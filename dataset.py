from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import io
import numpy as np
import os
import torch


from torch.utils.data import Dataset
from utils.utils import mu_law_encode

class CustomDataset(Dataset):

    def __init__(self,
                 meta_file,
                 receptive_field,
                 sample_size=5000,
                 upsample_factor=200,
                 quantization_channels=256,
                 use_local_condition=True,
                 noise_injecting=True,
                 feat_transform=None):
        with open(meta_file, encoding='utf-8') as f:
            self.metadata = [line.strip().split('|') for line in f]
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.upsample_factor = upsample_factor
        self.quantization_channels = quantization_channels
        self.use_local_condition = use_local_condition
        self.feat_transform = feat_transform
        self.noise_injecting = noise_injecting

        self.audio_buffer, self.local_condition_buffer = self._load_data(
                           self.metadata, use_local_condition, post_fn=lambda x: np.load(x))

    def __len__(self):
        return len(self.audio_buffer)

    def __getitem__(self, index):
        audios = self.audio_buffer[index]
        rand_pos = np.random.randint(0, len(audios[0]) - self.sample_size)

        if self.use_local_condition:
            local_condition = self.local_condition_buffer[index]
            local_condition = np.repeat(local_condition, self.upsample_factor, axis=0)
            local_condition = local_condition[rand_pos : rand_pos + self.sample_size]
        else:
            audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
            local_condition = None

        audio1=audios[0]
        audio1 = audio1[rand_pos : rand_pos + self.sample_size]
        target1 = mu_law_encode(audio1, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audio1.shape)
            audio1 = audio1 + noise
        audio1 = np.pad(audio1, [[self.receptive_field, 0], [0, 0]], 'constant')

        audio2 = audios[1]
        audio2 = audio2[rand_pos : rand_pos + self.sample_size]
        target2 = mu_law_encode(audio2, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audio2.shape)
            audio2 = audio2 + noise
        audio2 = np.pad(audio2, [[self.receptive_field, 0], [0, 0]], 'constant')

        audio3 = audios[2]
        audio3 = audio3[rand_pos : rand_pos + self.sample_size]
        target3 = mu_law_encode(audio3, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audio3.shape)
            audio3 = audio3 + noise
        audio3 = np.pad(audio3, [[self.receptive_field, 0], [0, 0]], 'constant')

        audio4 = audios[3]
        audio4 = audio4[rand_pos : rand_pos + self.sample_size]
        target4 = mu_law_encode(audio4, self.quantization_channels)
        if self.noise_injecting:
            noise = np.random.normal(0.0, 1.0/self.quantization_channels, audio4.shape)
            audio4 = audio4 + noise
        audio4 = np.pad(audio4, [[self.receptive_field, 0], [0, 0]], 'constant')
        
        audios=[]
        audios.append(audio1)
        audios.append(audio2)
        audios.append(audio3)
        audios.append(audio4)
        audios = np.array(audios)
        target=[]
        target.append(target1)
        target.append(target2)
        target.append(target3)
        target.append(target4)
        target = np.array(target)

        local_condition = np.pad(local_condition, [[self.receptive_field, 0], [0, 0]], 'constant')
        return torch.FloatTensor(audios), torch.LongTensor(target), torch.FloatTensor(local_condition)

    def _load_data(self, metadata, use_local_condition, post_fn=lambda x: x):
        audio_buffer = []
        local_condition_buffer = []
        for x in metadata:
            tmp_data1 = post_fn(x[0])
            tmp_data2 = post_fn(x[1])
            tmp_data3 = post_fn(x[2])
            tmp_data4 = post_fn(x[3])
            if len(tmp_data1) - self.sample_size - self.receptive_field > 0:
                tmp_data = []
                tmp_data.append(tmp_data1)
                tmp_data.append(tmp_data2)
                tmp_data.append(tmp_data3)
                tmp_data.append(tmp_data4)
                audio_buffer.append(tmp_data)
                if use_local_condition:
                    feat = post_fn(x[4])
                    if self.feat_transform is not None:
                        feat = self.feat_transform(feat)
                    local_condition_buffer.append(feat)
        return audio_buffer, local_condition_buffer

