from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tqdm
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class FFTNetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shift,
                 local_condition_channels=None,
                 first_band_input=False): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift = shift
        self.local_condition_channels = local_condition_channels
        self.first_band_input = first_band_input
        self.x1_l_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.x1_r_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        if local_condition_channels is not None:
            self.h_l_conv = nn.Conv1d(local_condition_channels, out_channels, kernel_size=1)
            self.h_r_conv = nn.Conv1d(local_condition_channels, out_channels, kernel_size=1)
        if first_band_input is True:
            self.x2_l_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.x2_r_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.output_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2=None, h=None):
        x1_l = self.x1_l_conv(x1[:, :, :-self.shift]) 
        x1_r = self.x1_r_conv(x1[:, :, self.shift:])
        if x2 is None:
            if h is None:
                z = F.relu(x1_l + x1_r)
                output = F.relu(self.output_conv(z) + x1_l +x1_r)
            else:
                h = h[:, :, -x1.size(-1):]
                h_l = self.h_l_conv(h[:, :, :-self.shift])
                h_r = self.h_r_conv(h[:, :, self.shift:])
                z_x = x1_l + x1_r
                z_h = h_l + h_r
                z = F.relu(z_x + z_h)
                output = F.relu(self.output_conv(z) + z_h + z_x)
        else:
            x2_l = self.x2_l_conv(x2[:, :, :-self.shift])
            x2_r = self.x2_r_conv(x2[:, :, self.shift:])
            if h is None:
                z = F.relu(x1_l + x1_r + x2_l + x2_r)
                output = F.relu(self.output_conv(z) + x_l +x_r)
            else:
                h = h[:, :, -x1.size(-1):]
                h_l = self.h_l_conv(h[:, :, :-self.shift])
                h_r = self.h_r_conv(h[:, :, self.shift:])
                z_x1 = x1_l + x1_r
                z_x2 = x2_l + x2_r
                z_h = h_l + h_r
                z = F.relu(z_x1 + z_x2 + z_h)
                output = F.relu(self.output_conv(z) + z_h + z_x1 + z_x2)

        return output


class FFTNet(nn.Module):
    """Implements the FFTNet for vocoder

    Reference: FFTNet: a Real-Time Speaker-Dependent Neural Vocoder. ICASSP 2018

    Args:
        n_stacks: the number of stacked fft layer
        fft_channels:
        quantization_channels:
        local_condition_channels:
    """
    def __init__(self, 
                 n_stacks=11, 
                 fft_channels=256, 
                 quantization_channels=256, 
                 local_condition_channels=None):
        super().__init__()
        self.n_stacks = n_stacks
        self.fft_channels = fft_channels
        self.quantization_channels = quantization_channels
        self.local_condition_channels = local_condition_channels
        self.window_shifts = [2 ** i for i in range(self.n_stacks)]
        self.receptive_field = sum(self.window_shifts) + 1
        self.linear = nn.Linear(fft_channels, quantization_channels)
        self.layers = nn.ModuleList()

        for shift in reversed(self.window_shifts):
            if shift == self.window_shifts[-1]:
                in_channels = 1
                fftlayer = FFTNetBlock(in_channels, fft_channels, shift, local_condition_channels, True)
            else:
                in_channels = fft_channels
                fftlayer = FFTNetBlock(in_channels, fft_channels, shift, None, False)
            self.layers.append(fftlayer)

    def forward(self, x1, x2, h):
        output = x1.transpose(1, 2)
        if x2 is not None:
            x2=x2.transpose(1,2)

        for fft_layer in self.layers:
            if fft_layer == self.layers[0]:
                output = fft_layer(output, x2, h)
            else:
                output = fft_layer(output, None, None)
        output = self.linear(output.transpose(1, 2))
        return output.transpose(1, 2)

class Subband_FFTNet(nn.Module):
    def __init__(self, 
             num_band=4,
             downsampling_factor=4,
             n_stacks=11, 
             fft_channels=256, 
             quantization_channels=256, 
             local_condition_channels=None):
        super().__init__()
        self.num_band=num_band
        self.downsampling_factor=downsampling_factor
        self.n_stacks=n_stacks
        self.window_shifts = [2 ** i for i in range(self.n_stacks)]
        self.receptive_field = sum(self.window_shifts) + 1
        self.quantization_channels = quantization_channels
        self.local_condition_channels = local_condition_channels
        self.linear = nn.Linear(fft_channels, quantization_channels)
        self.layers = nn.ModuleList()
        for i in range(num_band):
            self.layers.append(FFTNet(n_stacks, fft_channels, quantization_channels, local_condition_channels))

    def forward(self, x, h):
        for model in self.layers:
            if model == self.layers[0]:
                out1 = model(x[:,0], None, h)
            elif model == self.layers[1]:
                out2 = model(x[:,1], x[:,0], h)
            elif model == self.layers[2]:
                out3 = model(x[:,2], x[:,0], h)
            elif model == self.layers[3]:
                out4 = model(x[:,3], x[:,0], h)
        
        return torch.stack((out1,out2, out3, out4), 1)










	


        
