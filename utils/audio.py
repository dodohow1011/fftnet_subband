import librosa
import librosa.filters
import math
import pyworld
import pysptk
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.interpolate import interp1d
from hparams import hparams
import sys
import os
"""Reference:
    https://github.com/keithito/tacotron/blob/master/util/audio.py
    https://github.com/kan-bayashi/PytorchWaveNetVocoder/blob/master/src/bin/feature_extract.py
"""

def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]


def downsample(wav):
    return librosa.core.resample(wav, orig_sr=hparams.sample_rate, target_sr=hparams.downsample_rate)

def upsample(wav):
    return liborosa.core.resample(wav, orig_sr=4000, target_sr=16000)

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  librosa.output.write_wav(path, wav.astype(np.int16), hparams.sample_rate)


def preemphasis(x):
  return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S).T


def subband(y):
  # stft
  D = _stft(preemphasis(y))
  # size of exch band
  band = D.shape[0]
  # inverse ft
  n_fft, hop_length, win_length = _stft_parameters()
  wav1 = librosa.istft(D[:band//4,:], hop_length=hop_length, win_length=win_length)
  wav2 = librosa.istft(D[band//4:band//2,:], hop_length=hop_length, win_length=win_length)
  wav3 = librosa.istft(D[band//2:band*3//4,:], hop_length=hop_length, win_length=win_length)
  wav4 = librosa.istft(D[band*3//4:band,:], hop_length=hop_length, win_length=win_length)

  wav1 = downsample(wav1)
  wav2 = downsample(wav2)
  wav3 = downsample(wav3)
  wav4 = downsample(wav4)
  return wav1, wav2, wav3, wav4

def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length

def synthesis(sample_1, sample_2, sample_3, sample_4):
    wav1 = upsample(sample_1)
    wav2 = upsample(sample_2)
    wav3 = upsample(sample_3)
    wav4 = upsample(sample_4)

    D1 = _stft(preemphases(wav1))
    D2 = _stft(preemphases(wav2))
    D3 = _stft(preemphases(wav3))
    D4 = _stft(preemphases(wav4))

    D = np.vstack(D1, D2)
    D = np.vstack(D, D3)
    D = np.vstack(D, D4)

    n_fft, hop_length, win_length = _stft_parameters()
    wav = librosa.istft(D, hop_length=hop_length, win_length=win_length)

    return wav
    
    
# Not tested
def extract_mcc(wav):
    wav = np.array(wav, dtype=np.float)
    n_fft = (hparams.num_freq - 1) * 2
    f0, time_axis = pyworld.harvest(wav, hparams.sample_rate, f0_floor=hparams.minf0,
                                    f0_ceil=hparams.maxf0, frame_period=hparams.frame_shift_ms)
    spc = pyworld.cheaptrick(wav, f0, time_axis, hparams.sample_rate, fft_size=n_fft)

    f0[f0 < 0] = 0
    uv, cont_f0 = convert_continuos_f0(f0)

    mcep = pysptk.sp2mc(spc, hparams.mcep_dim, alpha=hparams.mcep_alpha)
    uv = np.expand_dims(uv, axis=-1)
    cont_f0 = np.expand_dims(cont_f0, axis=-1)
    feats = np.concatenate([uv, cont_f0, mcep], axis=1)
    return feats

def convert_continuos_f0(f0):
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)
