import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matlab.engine

class SendOverChannel(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tx_symbols, rx_symbols, channel_matrix):
            ctx.save_for_backward(channel_matrix)
            return torch.cat(rx_symbols.real, rx_symbols.imag)
        
        @staticmethod
        def backward(ctx, grad_output):
            channel_matrix = ctx.saved_tensors
            (grad_output_real, grad_output_imag) = grad_output.chunk(2)
            grad_input_real = grad_output_real @ channel_matrix.real.T - grad_output_imag @ channel_matrix.imag.T
            grad_input_imag = grad_output_real @ channel_matrix.imag.T + grad_output_imag @ channel_matrix.real.T
            grad_input = torch.cat(grad_input_real, grad_input_imag)
            return grad_input, None, None

class MIMOChannel(nn.Module):
    def __init__(self, n_streams, PSNR=20.0, model="rayleigh") -> None:
        super().__init__()
        self.N_S = n_streams
        self.PSNR = PSNR
        self.p = 1
        self.register_buffer('channel_matrix', torch.complex(torch.randn(self.N_S, self.N_S), torch.randn(self.N_S, self.N_S)))

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
        self.channel_matrix = torch.complex(torch.randn(self.N_S, self.N_S), torch.randn(self.N_S, self.N_S))

    def update_channel_matrix(self):
        if self.eng is None:
            assert len(matlab.engine.find_matlab()) > 0, "No running instance of MATLAB is found"
            self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        # Send NDP packet
        # rx_samples_matlab, CFRs_matlab = self.eng.mimoPhyDnn(np.random.randn(1, self.N_S), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
        CFRs_matlab = self.eng.MIMODistNet_OFDM(self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=1)
        self.channel_matrix = torch.tensor(CFRs_matlab)

    def forward(self, x):
        if self.model == "awgn":
            return x + torch.randn_like(x) * self.noise_scale
        elif self.model == "rayleigh":
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            original_shape = x.shape
            tx_symbols = torch.complex(*x.reshape(-1, self.N_S).chunk(2))
            rx_symbols = F.linear(tx_symbols, self.channel_matrix)
            x = torch.cat(rx_symbols.real, rx_symbols.imag).reshape(original_shape)
            return x + torch.randn_like(x) * self.noise_scale
        elif self.model == "pi-radio":
            assert x.numel() % self.N_S == 0, f"Tensor of length {x.numel} cannot be divided into {self.N_S} parts for different data streams"
            original_shape = x.shape
            tx_symbols = torch.complex(*x.reshape(-1, self.N_S).chunk(2))
            rx_symbols_matlab, CFRs_matlab = self.eng.MIMODistNet_OFDM(tx_symbols.detach().cpu().numpy(), self.eng.workspace['sdr0'], self.eng.workspace['sdr1'], nargout=2)
            self.channel_matrix = torch.tensor(CFRs_matlab)
            rx_symbols = torch.tensor(rx_symbols_matlab)
            x = SendOverChannel.apply(tx_symbols, rx_symbols, self.channel_matrix)
            x.reshape(original_shape)
