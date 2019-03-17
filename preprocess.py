from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import fnmatch
import numpy as np
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from hparams import hparams
from utils import audio
import sys
import os

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def _process_wav(wav_path, audio_path, spc_path):
    wav = audio.load_wav(wav_path)
    wav1, wav2, wav3, wav4 = audio.subband(wav)

    if hparams.feature_type == 'mcc':
        # Extract mcc and f0
        spc = audio.extract_mcc(wav)
    else:
        # Extract mels
        spc = audio.melspectrogram(wav).astype(np.float32)

    # Align audios and mels
    hop_length = int(hparams.frame_shift_ms / 4000 * hparams.sample_rate)
    length_diff_1 = len(spc) * hop_length - len(wav1)
    length_diff_2 = len(spc) * hop_length - len(wav2)
    length_diff_3 = len(spc) * hop_length - len(wav3)
    length_diff_4 = len(spc) * hop_length - len(wav4)
    wav1 = wav1.reshape(-1,1)
    if length_diff_1 > 0:
        wav1 = np.pad(wav1, [[0, length_diff_1], [0, 0]], 'constant')
    elif length_diff_1 < 0:
        wav1 = wav1[: hop_length * spc.shape[0]]
    wav2 = wav2.reshape(-1,1)
    if length_diff_2 > 0:
        wav2 = np.pad(wav2, [[0, length_diff_2], [0, 0]], 'constant')
    elif length_diff_2 < 0:
        wav2 = wav2[: hop_length * spc.shape[0]]
    wav3 = wav3.reshape(-1,1)
    if length_diff_3 > 0:
        wav3 = np.pad(wav1, [[0, length_diff_3], [0, 0]], 'constant')
    elif length_diff_3 < 0:
        wav3 = wav3[: hop_length * spc.shape[0]]
    wav4 = wav4.reshape(-1,1)
    if length_diff_4 > 0:
        wav4 = np.pad(wav4, [[0, length_diff_4], [0, 0]], 'constant')
    elif length_diff_4 < 0:
        wav4 = wav4[: hop_length * spc.shape[0]]
    fid1 = os.path.basename(audio_path).replace('.npy', '_band1.npy')
    fid2 = os.path.basename(audio_path).replace('.npy', '_band2.npy')
    fid3 = os.path.basename(audio_path).replace('.npy', '_band3.npy')
    fid4 = os.path.basename(audio_path).replace('.npy', '_band4.npy')

    fid1 = os.path.join('training_data/audios', fid1)
    
    fid2 = os.path.join('training_data/audios', fid2)
    fid3 = os.path.join('training_data/audios', fid3)
    fid4 = os.path.join('training_data/audios', fid4)
    
    np.save(fid1, wav1)
    np.save(fid2, wav2)
    np.save(fid3, wav3)
    np.save(fid4, wav4)
    np.save(spc_path, spc)
    return (fid1, fid2, fid3, fid4, spc_path, spc.shape[0])


def calc_stats(file_list, out_dir):
    scaler = StandardScaler()
    for i, filename in enumerate(file_list):
        feat = np.load(filename)
        scaler.partial_fit(feat)

    mean = scaler.mean_
    scale = scaler.scale_
    if hparams.feature_type == "mcc": 
        mean[0] = 0.0
        scale[0] = 1.0
    
    np.save(os.path.join(out_dir, 'mean'), np.float32(mean))
    np.save(os.path.join(out_dir, 'scale'), np.float32(scale))


def build_from_path(in_dir, audio_out_dir, mel_out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wav_list = find_files(in_dir)
    for wav_path in wav_list:
        fid = os.path.basename(wav_path).replace('.wav','.npy')
        audio_path = os.path.join(audio_out_dir, fid)
        mel_path = os.path.join(mel_out_dir, fid)
        futures.append(executor.submit(partial(_process_wav, wav_path, audio_path, mel_path)))

    return [future.result() for future in tqdm(futures)]
    

def preprocess(args):
    in_dir = os.path.join(args.wav_dir)
    out_dir = os.path.join(args.output)
    audio_out_dir = os.path.join(out_dir, 'audios')
    mel_out_dir = os.path.join(out_dir, 'mels')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, audio_out_dir, mel_out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

    spc_list = find_files(mel_out_dir, "*.npy")
    calc_stats(spc_list, out_dir)
     

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[5] for m in metadata])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='cmu_us_slt_arctic/wav')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
