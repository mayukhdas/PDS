import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from statistics import mean
from torch.nn import MSELoss
from torchmetrics.functional.clustering import davies_bouldin_score
from torch_clustering import PyTorchKMeans
from accelerate import Accelerator
import h5py
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import gym

accelerator = Accelerator()

class HDF5Dataset(Dataset):
    def __init__(self, file_path, window_size=5):

        self.file_path = file_path
        self.window_size = window_size

        with h5py.File(self.file_path, 'r') as f:
            self.observations = f['observations'][:]
            self.actions = f['actions'][:]

        self.length = len(self.observations)
        self.obs_dim = self.observations.shape[1]
        self.act_dim = self.actions.shape[1]

        # Create next_observations
        self.next_observations = np.roll(self.observations, -1, axis=0)
        self.next_observations[-1] = self.observations[-1]  # Set last next_observation to be the same as the last observation
        ##########################
        self.mean = np.mean(self.observations, axis=0)
        self.std = np.std(self.observations, axis=0)
        # normalize the observations
        print("mean", self.mean)
        print("std", self.std)
        self.observations = (self.observations - self.mean) / self.std
        self.next_observations = (self.next_observations - self.mean) / self.std

    def __len__(self):
        return self.length - self.window_size + 1  # Adjust length to account for sliding window

    def __getitem__(self, idx):
        sa_hist = []
        for i in range(self.window_size):
            observation = self.observations[idx + i]
            action = self.actions[idx + i]
            sa = np.concatenate([action, observation])  # Note: Changed order to match your description
            sa_hist.append(sa)

        sa_hist = np.stack(sa_hist, axis=0)
        next_observation = self.next_observations[idx + self.window_size - 1]

        return torch.FloatTensor(sa_hist), torch.FloatTensor(next_observation)


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()

class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [input_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.matmul(x, self.dense_matrix)

class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[1]
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out, seq_len)
        return enc_out, dec_out


def train_model(
    model, train_loader, optimizer, verbose, lr, epochs, clip_value, n_clusters, save_path=None, clusterer=None
):
    criterion = MSELoss(reduction="mean")

    # Initialize lists to store metrics
    mean_losses = []
    mean_recon_losses = []
    mean_db_scores = []

    # Initialize TensorBoard writer
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(save_path, 'logs'))

    for epoch in range(1, epochs + 1):
        model.train()

        losses = []
        scores = []
        reconstruction_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch, _ in progress_bar:
            optimizer.zero_grad()
            # Forward pass
            embed, x_prime = model(batch)
 
            with torch.no_grad():
                cluster_labels = clusterer.fit_predict(embed)
                db_score = davies_bouldin_score(embed, cluster_labels)
                scores.append(db_score.item())

            recon_loss = criterion(x_prime, batch)
            recon_loss = recon_loss / batch.size(0) if criterion.reduction == 'sum' else recon_loss

            reconstruction_losses.append(recon_loss.item())

            loss = recon_loss * db_score

            # Backward pass
            accelerator.backward(loss)

            # Gradient clipping on norm
            if clip_value is not None:
                accelerator.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon_loss': f"{recon_loss.item():.4f}",
                'db_score': f"{db_score.item():.4f}"
            })

        mean_loss = mean(losses)
        mean_recon_loss = mean(reconstruction_losses)
        mean_db_score = mean(scores)
        
        mean_losses.append(mean_loss)
        mean_recon_losses.append(mean_recon_loss)
        mean_db_scores.append(mean_db_score)

        # Log metrics to TensorBoard
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train', mean_loss, epoch)
            writer.add_scalar('Reconstruction Loss/train', mean_recon_loss, epoch)
            writer.add_scalar('Davies-Bouldin Score/train', mean_db_score, epoch)

        if verbose:
            print(f"Epoch: {epoch}/{epochs}")
            print(f"  Average Loss: {mean_loss:.4f}")
            print(f"  Average Reconstruction Loss: {mean_recon_loss:.4f}")
            print(f"  Average Davies-Bouldin Score: {mean_db_score:.4f}")

        if save_path and (epoch % 20 == 0):
            accelerator.save(model.state_dict(), f"{save_path}/lstm_ae_model_epoch_{epoch}.pt")

    # Close the TensorBoard writer
    if accelerator.is_main_process:
        writer.close()

    return mean_losses, mean_recon_losses, mean_db_scores

def main(env_name, dataset_path):
    # Hyperparameters
    batch_size = 16000
    lr = 1e-3
    epochs = 100
    clip_value = 1
    n_clusters = 10
    verbose = True
    window_size = 5

    save_path = f'trained_models_new/{env_name}_trial_final/'
    
    dataset = HDF5Dataset(dataset_path, window_size=window_size)
    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    input_dim = ((obs_dim + act_dim), window_size)  # input_dim shape -> (observation_dim + action_dim) , window_size
    model = LSTM_AE(
        input_dim=input_dim[0],
        encoding_dim=16,
        h_dims=[32],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initialize the clusterer
    kwargs = {
        'metric': 'euclidean',
        'distributed': False,
        'random_state': 0,
        'n_clusters': n_clusters,
        'verbose': False
    }
    old_centers = None
    if old_centers is None:
        clusterer = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    else:
        clusterer = PyTorchKMeans(init=old_centers, max_iter=300, tol=1e-4, **kwargs)

    model, train_loader, optimizer, clusterer = accelerator.prepare(model, train_loader, optimizer, clusterer)

    # create the save directory if it doesn't exist
    if accelerator.is_local_main_process:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)    

    # Train the model
    losses, recon_losses, db_scores = train_model(
        model, train_loader, optimizer, verbose, lr, epochs, clip_value, n_clusters, save_path, clusterer
    )

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f"{save_path}/lstm_ae_model_final.pt")

    print("Training completed. Final model saved to", f"{save_path}/lstm_ae_model_final.pt")

    # Plot and save the metrics
    if accelerator.is_main_process:
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, epochs + 1), losses, label='Total Loss')
        plt.plot(range(1, epochs + 1), recon_losses, label='Reconstruction Loss')
        plt.plot(range(1, epochs + 1), db_scores, label='Davies-Bouldin Score')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM AE model")
    parser.add_argument("--env_name", type=str, required=True, help="Name of the gym environment")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file")
    args = parser.parse_args()

    main(args.env_name, args.dataset_path)