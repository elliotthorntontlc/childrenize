import numpy as np
import soundfile as sf
import pyworld as pw

import argparse
import sw

import scipy.signal
import scipy.ndimage

EPSILON = 1e-8
F0_MAX = 600
F0_MIN = 50
GENDER_F0_THRESHOLD = 160

parser = argparse.ArgumentParser()
parser.add_argument("input_audio", help="Path of input audio.")
parser.add_argument("output_audio", help="Path of output audio.")
parser.add_argument("-t", "--vowel_stretch_factor", default=None, help='Vowel time stretching factor.')
parser.add_argument("-f", "--target_f0", default=None, help="Target F0.")
parser.add_argument("-s", "--warping_factor", default=None, help="Spectral warping factor.")

def reduce_clicks_pops(sig, kernel_size=3):
    # Apply a median filter to reduce short impulsive noise (clicks/pops)
    if sig.ndim == 1:
        return scipy.signal.medfilt(sig, kernel_size=kernel_size)
    else:
        # For stereo or multi-channel, apply per channel
        return np.stack([scipy.signal.medfilt(sig[:, ch], kernel_size=kernel_size) for ch in range(sig.shape[1])], axis=1)

def process(sig, fs,
            target_f0=None,
            warping_function='linear',
            warping_factor=1,
            vowel_stretch_factor=1):

    _f0_h, t_h = pw.harvest(sig, fs)
    f0_h = pw.stonemask(sig, _f0_h, t_h, fs)
    # Optional: f0_h = scipy.signal.medfilt(f0_h, kernel_size=3)  # Less smoothing

    sp_h = pw.cheaptrick(sig, f0_h, t_h, fs)
    # Optional: sp_h = scipy.ndimage.gaussian_filter1d(sp_h, sigma=0.3, axis=1)  # Less smoothing

    ap_h = pw.d4c(sig, f0_h, t_h, fs)
    # Optional: ap_h = np.clip(ap_h, 0, 1.0)  # Less aggressive clamping

    f0_mean = np.mean(f0_h[f0_h > F0_MIN])

    if target_f0 is None:
        target_f0 = f0_mean

    f0_shift = target_f0 - f0_mean
    f0_out = np.copy(f0_h)
    f0_out[f0_h > F0_MIN] = f0_h[f0_h > F0_MIN] + f0_shift
    f0_out[f0_out > F0_MAX] = F0_MAX

    assert warping_function in ['linear', 'piecewise']
    sp_out = sw.transform(sp_h, warping_factor, fs, warping_function)

    harmonics = f0_h != 0
    change_points = np.nonzero(harmonics != np.roll(harmonics, 1))[0] + 1
    if len(change_points) > 1 and change_points[-1] >= len(harmonics):
        change_points = change_points[:-1]

    is_vowel = harmonics[np.insert(change_points, 0, 0)]
    n_segs = len(is_vowel)
    sp_out_segs = np.split(sp_out, change_points)
    ap_segs = np.split(ap_h, change_points)
    f0_out_segs = np.split(f0_out, change_points)
    y_segs = []

    frame_period = 5.0  # ms, closer to default WORLD

    for i in range(n_segs):
        if is_vowel[i]:
            sig = pw.synthesize(f0_out_segs[i], sp_out_segs[i], ap_segs[i], fs, vowel_stretch_factor * frame_period)
        else:
            sig = pw.synthesize(f0_out_segs[i], sp_out_segs[i], ap_segs[i], fs, frame_period)
        y_segs.append(sig)

    y_h = np.concatenate(y_segs)

    # Remove Wiener filter for now
    # y_h = scipy.signal.wiener(y_h)

    return y_h

def randomize_parameters(sig, fs,
                         target_f0=None,
                         warping_function=None,
                         warping_factor=None,
                         vowel_stretch_factor=None):

    params = {}
    if target_f0 is None:
        params['target_f0'] = np.random.uniform(240, 300)
    else:
        params['target_f0'] = float(target_f0)

    if vowel_stretch_factor is None:
        params['vowel_stretch_factor'] = np.random.uniform(1.1, 1.4)
    else:
        params['vowel_stretch_factor'] = float(vowel_stretch_factor)

    if warping_function is None:
        _f0_h, t_h = pw.harvest(sig, fs)
        f0_h = pw.stonemask(sig, _f0_h, t_h, fs)
        f0_mean = np.mean(f0_h[f0_h > 50])

        if f0_mean < GENDER_F0_THRESHOLD:  # Assuming the utterance to be male adult
            params['warping_function'] = 'linear'
            if warping_factor is None:
                params['warping_factor'] = np.random.uniform(1.2, 1.4)
            else:
                params['warping_factor'] = float(warping_factor)

        else:
            params['warping_function'] = 'piecewise'
            if warping_factor is None:
                params['warping_factor'] = np.random.uniform(1.1, 1.25)
            else:
                params['warping_factor'] = float(warping_factor)

    return params

def main():
    args = parser.parse_args()
    filename = args.input_audio
    x, fs = sf.read(filename)
    # --- Click/pop reduction step ---
    x = reduce_clicks_pops(x, kernel_size=3)
    params = randomize_parameters(x, fs, target_f0=args.target_f0, warping_factor=args.warping_factor, vowel_stretch_factor=args.vowel_stretch_factor)
    print(params)
    y = process(x, fs, **params)
    sf.write(args.output_audio, y, fs)
    return

if __name__ == '__main__':
    main()
