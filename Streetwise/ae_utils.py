import torch
import torch.nn as nn
import csv
import json
import numpy as np
import scipy
from scipy import optimize

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def scale_mse_loss(mse_loss, min_new=0.01, max_new=0.1, steepness=10):
#     # Apply log transformation to compress range
#     # log_loss = torch.log1p(mse_loss)  # log1p(x) = log(1 + x)
#     log_loss = torch.clamp((mse_loss), 0, 2)
    
#     # Normalize to [0, 1] range
#     # normalized = (log_loss - log_loss.min()) / (log_loss.max() - log_loss.min())
#     normalized = log_loss / log_loss.max()
    
#     # Apply sigmoid-like function for polarization
#     sigmoid = 1 / (1 + torch.exp(-steepness * (normalized - 0.5)))
    
#     # Scale to desired range
#     scaled = sigmoid * (max_new - min_new) + min_new
    
#     return scaled

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
        enc_out = enc_out.unsqueeze(0)
        dec_out = self.decoder(enc_out, seq_len)
        return enc_out, dec_out


class AEOpexCellStateModel(nn.Module):
    def __init__(self, policy, ae, qf, num_ae_features=23, mean=None, std_dev=None, beta=0.1, gamma=0.1, act_dim=6):
        super().__init__()
        self.policy = policy.to(DEFAULT_DEVICE)
        self.ae_model = ae.to(DEFAULT_DEVICE)
        self.qf = qf.to(DEFAULT_DEVICE)
        self.ae_timesteps = 5
        self.ae_features = num_ae_features
        self.mse_loss_history = []
        self.mse_grad_history = []
        self.grad_a_history = []

        self.mean = mean
        self.std_dev = std_dev

        self.beta = beta
        self.gamma = gamma
        self.act_dim = act_dim


    def get_mse_loss_history(self):
        return np.array(self.mse_loss_history)

    def forward(self, state, hidden_state, cell_state, deterministic=True):
        state = state.unsqueeze(0)
        action = self.policy.act(state, enable_grad=True, deterministic=deterministic)

        grad_a = self.calc_q_grad(state.squeeze(0), action.squeeze(0))

        ae_history = cell_state.view(-1, self.ae_timesteps, self.ae_features).to(action.device)

        norm_state = (state.squeeze(0) - torch.tensor(self.mean).to(action.device)) / torch.tensor(self.std_dev).to(action.device)
        norm_state = norm_state.unsqueeze(0)
        new_obs = torch.reshape(torch.cat([action, norm_state], dim=-1), (1,1,-1))
        ae_history = torch.cat([ae_history[:, 1:, :], new_obs], dim=1)

        next_cell_state = ae_history

        _, ae_output = self.ae_model(ae_history)


        mse_loss = torch.nn.functional.mse_loss(ae_output, ae_history, reduction='none')
        
        mse_loss_first_17 = mse_loss.mean(dim=(1), keepdim=True).squeeze(1)[:,:-self.act_dim].mean(dim=(1), keepdim=True).squeeze(1)
        mse_loss_last_6 = mse_loss.mean(dim=(1), keepdim=True).squeeze(1)[:,-self.act_dim:]

        mse_loss = mse_loss_first_17 * mse_loss_last_6

        min_old, max_old = 0, 2
        min_new, max_new = 0.01, 0.1

        clamped_mse_loss = torch.clamp(torch.exp(mse_loss), min_old, max_old)
        scaled_mse_loss = (clamped_mse_loss - min_old) / (max_old - min_old) * (max_new - min_new) + min_new

        # scaled_mse_loss = scale_mse_loss(mse_loss, min_new=0.01, max_new=0.1, steepness=10)
        
    
        # updated_action = action + (torch.clamp((mse_loss), 0.01, 0.1) * grad_a).to(device=action.device)
        updated_action = action + (scaled_mse_loss * grad_a).to(device=action.device)

        # updated_action = action
        # updated_action = action + 0.1 * grad_a
        # updated_action = action + 0.01 * grad_a

        # Store the MSE loss
        self.mse_loss_history.extend(mse_loss.detach().cpu().numpy().tolist())
        # self.mse_grad_history.extend(mse_grad.detach().cpu().numpy().tolist())
        self.grad_a_history.extend(grad_a.detach().cpu().numpy().tolist())
        

        return updated_action, hidden_state, next_cell_state

    def calc_q_grad(self, state, action):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        qf_output = self.qf(state, action, enable_grad=True)
        return torch.autograd.grad(qf_output, action, create_graph=True, retain_graph=True)[0]

    # function to calculate the gradient of MSE loss w.r.t. action
    def calc_mse_grad(self, mse_loss, action):
        return torch.autograd.grad(mse_loss, action, create_graph=True, retain_graph=True)[0]