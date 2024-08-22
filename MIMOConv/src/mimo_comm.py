import math

import torch
import torch.nn as nn

class MIMOChannel(nn.Module):
    def __init__(self, n_streams, PSNR, model="rayleigh") -> None:
        super().__init__()
        self.N_S = n_streams
        self.PSNR = PSNR
        self.p = 1
        self.register_buffer('channel_matrix', torch.randn(self.N_S, self.N_S))

        self.noise_scale = math.sqrt(0.5 * self.p / math.pow(10, self.PSNR / 10))
        self.model = model
    
    def set_model(self, model):
        assert model in ["awgn", "rayleigh"], "MIMO channel model is not supported"
        self.model = model
    
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def randomize_channel_matrix(self):
        self.channel_matrix = torch.randn(self.N_S, self.N_S)

    def forward(self, x):
        if self.model == "awgn":
            return x + torch.randn_like(x) * self.noise_scale
        elif self.model == "rayleigh":
            assert x.shape[0] % self.N_S == 0, f"Signal of length {x.shape[0]} cannot be divided into {self.N_S} streams"
            return (self.channel_matrix @ x.reshape(self.N_S, -1)).reshape(x.shape) + torch.randn_like(x) * self.noise_scale

