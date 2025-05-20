import os
import subprocess
import sys
import numpy as np

input_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.lower().endswith('.wav'):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        print(f"Processing {in_path} -> {out_path}")
        subprocess.run([
            sys.executable,  # Use the current Python interpreter
            "childrenize.py",
            in_path,
            out_path,
            "-t", "1.1",
            "-f", "288",
            "-s", "1.2"
            # "-t", "1.0",
            # "-f", "240",
            # "-s", "1.05"
            # # -t: vowel time stretch factor
            # "-t", "1.05",
            # # -f: target f0
            # "-f", "288",
            # # -s: warping factor
            # "-s", "1.23"
        ])

        # Additional code for post-processing with smoothing (low-pass filter)
        # The gist: a 6th-order Butterworth low-pass filter with a cutoff frequency of 4000 Hz
        from scipy.io import wavfile
        from scipy.signal import butter, filtfilt

        def lowpass_filter(data, sr, cutoff=4000, order=6):
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)

        sr, data = wavfile.read(out_path)
        # Only apply to mono or stereo int16 files
        if data.dtype == np.int16:
            # Normalize to float
            data_f = data.astype(np.float32) / 32768.0
            # If stereo, filter both channels
            if data_f.ndim == 2:
                data_filt = np.stack([lowpass_filter(ch, sr) for ch in data_f.T], axis=1)
            else:
                data_filt = lowpass_filter(data_f, sr)
            # Back to int16
            data_out = np.clip(data_filt * 32768, -32768, 32767).astype(np.int16)
            wavfile.write(out_path, sr, data_out)