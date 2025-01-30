import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matlab.engine

class MIMOChannel(nn.Module):
    def __init__(self, n_streams, PSNR=20.0, model="rayleigh") -> None:
        super().__init__()
        self.N_S = n_streams
        self.PSNR = PSNR
        self.p = 1
        self.register_buffer('channel_matrix', torch.randn(self.N_S, self.N_S))

        self.noise_scale = math.sqrt(0.5 * self.p / math.pow(10, self.PSNR / 10))
        self.model = model

        self.eng = None
        if self.model == "pi-radio":
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
    
    def set_model(self, model):
        assert model in ["awgn", "rayleigh", "pi-radio"], "MIMO channel model is not supported"
        self.model = model

        if self.model == "pi-radio":
            if self.eng is not None:
                self.eng.quit()
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
    
    def set_channel_matrix(self, channel_matrix):
        self.channel_matrix = channel_matrix
    
    def randomize_channel_matrix(self):
        self.channel_matrix = torch.randn(self.N_S, self.N_S)

    def update_channel_matrix(self):
        if self.eng is None:
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        # Send NDP packet
        # rx_samples_matlab, CFRs_matlab = self.eng.mimoPhyDnn(np.random.randn(1, self.N_S), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
        CFRs_matlab = self.eng.MIMODistNet_OFDM(self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=1)
        self.channel_matrix = torch.tensor(CFRs_matlab).real

    def forward(self, x):
        if self.model == "awgn":
            return x + torch.randn_like(x) * self.noise_scale
        elif self.model == "rayleigh":
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            return F.linear(x.reshape(-1, self.N_S), self.channel_matrix).reshape(x.shape) + torch.randn_like(x) * self.noise_scale
        elif self.model == "pi-radio":
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            rx_samples_matlab, CFRs_matlab = self.eng.mimoPhyDnn(x.reshape(-1, self.N_S).detach().cpu().numpy(), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
            self.channel_matrix = torch.tensor(CFRs_matlab)[:, :, -1]
            return torch.tensor(rx_samples_matlab)