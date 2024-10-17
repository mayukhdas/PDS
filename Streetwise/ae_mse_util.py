import torch
import torch.nn as nn
import csv
import json
import numpy as np
import scipy
from scipy import optimize

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    

class LSTM_AE_EVAL(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE_EVAL, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[1]
        enc_out = self.encoder(x)
        enc_out = enc_out.unsqueeze(0)
        dec_out = self.decoder(enc_out, seq_len)
        return enc_out, dec_out


class AEOpexCellStateModelMSE(nn.Module):
    def __init__(self, policy, ae, qf, num_ae_features=23):
        super().__init__()
        self.policy = policy.to(DEFAULT_DEVICE)
        self.ae_model = ae.to(DEFAULT_DEVICE)
        self.qf = qf.to(DEFAULT_DEVICE)
        self.ae_timesteps = 5
        self.ae_features = num_ae_features
        self.mse_loss_history = []
        self.mse_grad_history = []
        self.grad_a_history = []
        self.mean = [-0.11004163,0.15697056,0.10378583,  0.14686427,  0.07841891, -0.20104693,-0.08223713, -0.2802379,   4.463383,   -0.07576275, -0.09257912,  0.4192936,-0.41244322,  0.11662842,-0.05935472, -0.09750155, -0.14589772]

        self.std_dev = [ 0.10859901,  0.6113909,   0.4913569,   0.44862902,  0.3971792,   0.48135453,  0.30596346,  0.26374257,  1.9015079,   0.93880796, 1.6241817,  14.426355,11.995561,   11.984273,   12.15909,     8.126285,    6.4183607 ]


    def get_mse_loss_history(self):
        return np.array(self.mse_loss_history)

    def forward(self, state, hidden_state, cell_state, deterministic=True):
        state = state.unsqueeze(0)
        action = self.policy.act(state, enable_grad=True, deterministic=deterministic)

        ae_history = cell_state.view(-1, self.ae_timesteps, self.ae_features).to(action.device)

        norm_state = (state.squeeze(0) - torch.tensor(self.mean).to(action.device)) / torch.tensor(self.std_dev).to(action.device)
        norm_state = norm_state.unsqueeze(0)
        new_obs = torch.reshape(torch.cat([action, state], dim=-1), (1,1,-1))
        ae_history = torch.cat([ae_history[:, 1:, :], new_obs], dim=1)

        next_cell_state = ae_history

        embed, ae_output = self.ae_model(ae_history)

        mse_loss = torch.nn.functional.mse_loss(ae_output, ae_history)

        mse_loss_grad_a = torch.nn.functional.mse_loss(ae_output, ae_history, reduction='none').mean(dim=(1, 2), keepdim=True).squeeze(1)
        grad_a = self.calc_q_grad(state.squeeze(0), action.squeeze(0), mse_loss_grad_a)

        # updated_action = action +  grad_a * 0.09
        # updated_action = action + torch.clamp((torch.log1p(mse_loss)) * grad_a * 0.2, -1, 1).to(action.device)
        mse_grad = self.calc_mse_grad(mse_loss, action)

        updated_action = action
        # updated_action = action - 3*mse_grad

        # Store the MSE loss
        self.mse_loss_history.append(mse_loss.item())
        self.mse_grad_history.extend(mse_grad.detach().cpu().numpy().tolist())
        self.grad_a_history.extend(grad_a.detach().cpu().numpy().tolist())
        

        return updated_action, hidden_state, next_cell_state

    def calc_q_grad(self, state, action, mse_loss):
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        qf_output = self.qf(state, action, mse_loss, enable_grad=True)
        return torch.autograd.grad(qf_output, action, create_graph=True, retain_graph=True)[0]

    # function to calculate the gradient of MSE loss w.r.t. action
    def calc_mse_grad(self, mse_loss, action):
        return torch.autograd.grad(mse_loss, action, create_graph=True, retain_graph=True)[0]