from pathlib import Path
from typing import Tuple, Union

import numpy as np
import h5py
import librosa
import soundfile as sf
from scipy import interpolate
from deep_demuffle.data_utils import AudioSegmentDual
from scipy.signal import decimate
from matplotlib import pyplot as plt


def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        print('List of arrays in input file:', list(hf.keys()))
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
        print('Shape of X:', X.shape)
        print('Shape of Y:', Y.shape)

    return X, Y


def load_audio_dataset(
        raw_audio_dir: Union[str, Path],
        muffled_audio_dir: Union[str, Path],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param raw_audio_dir:
    Directory with raw mp3 files
    :param muffled_audio_dir:
    Directory with muffled wav files
    :return:
    features, labels
    """

    raw_fps = sorted(Path(raw_audio_dir).glob('*.mp3'))
    muffled_fps = sorted(Path(muffled_audio_dir).glob('*.wav'))

    raw_names = {fp.stem for fp in raw_fps}
    muffled_names = {fp.stem for fp in muffled_fps}

    raw_names_not_found = muffled_names - raw_names
    muffled_names_not_found = raw_names - muffled_names

    assert not raw_names_not_found, f'Not found files in raw data folder: {raw_names_not_found}'
    assert not muffled_names_not_found, f'Not found files in muffled data folder: {muffled_names_not_found}'

    features = []
    labels = []

    for raw_fp, muffled_fp in zip(raw_fps, muffled_fps):

        raw_audio = AudioSegmentDual.from_mp3(raw_fp)
        muddled_audio = AudioSegmentDual.from_wav(muffled_fp)

        features.append(raw_audio.to_numpy())
        labels.append(muddled_audio.to_numpy())

    return stack_with_padding(features), stack_with_padding(labels)


def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    #x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp


def upsample_wav(wav, args, model, export_spectrum=False):
    # load signal
    x_hr, fs = librosa.load(wav, sr=args.sr)
    x_lr_t = decimate(x_hr, args.r)
    # pad to mutliple of patch size to ensure model runs over entire sample
    x_hr = np.pad(x_hr, (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 'constant', constant_values=(0,0))
    # downscale signal
    x_lr = decimate(x_hr, args.r)

    # upscale the low-res version
    x_lr = x_lr.reshape((1, len(x_lr), 1))

    # preprocessing
    assert len(x_lr) == 1
    x_sp = spline_up(x_lr, args.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(args.layers+1)))]
    x_sp = x_sp.reshape((1,len(x_sp),1))
    x_sp = x_sp.reshape((int(x_sp.shape[1]/args.patch_size), args.patch_size,1))

    # prediction
    pred = model.predict(x_sp, batch_size=16)
    x_pr = pred.flatten()

    # crop so that it works with scaling ratio
    x_hr = x_hr[:len(x_pr)]
    x_lr_t = x_lr_t[:len(x_pr)]

    # save the file
    outname = wav # + '.' + args.out_label
    sf.write(outname + '.lr.wav', x_lr_t, int(fs / args.r))
    sf.write(outname + '.hr.wav', x_hr, fs)
    sf.write(outname + '.pr.wav', x_pr, fs)

    if export_spectrum:
        # save the spectrum
        S = get_spectrum(x_pr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.pr.png')
        S = get_spectrum(x_hr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.hr.png')
        S = get_spectrum(x_lr, n_fft=int(2048/args.r))
        save_spectrum(S, outfile=outname + '.lr.png')


def get_spectrum(x, n_fft=2048):
    S = librosa.stft(x, n_fft)
    #p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S


def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    plt.savefig(outfile)


def stack_with_padding(arrays):
    changing_dim = 1

    max_len = max(array.shape[changing_dim] for array in arrays)
    for i, array in enumerate(arrays):
        pad = np.zeros((2, max_len-array.shape[changing_dim]))

        array = np.concatenate((array, pad), axis=changing_dim)
        arrays[i] = array
    return np.stack(arrays)
