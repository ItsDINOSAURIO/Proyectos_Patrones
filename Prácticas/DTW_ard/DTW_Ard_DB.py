# import numpy as np
# import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
# import csv

# # Prototype audio file paths
# proto_files = [
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\zero_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\uno_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\dos_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\tres_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\cuatro_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\cinco_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\seis_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\siete_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\ocho_01.wav",
#     r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\nueve_01.wav"
# ]

# # Store prototype features
# proto_feats = []

# # Window parameters
# win_dur = 0.02  # 20 ms
# alpha = 0.95  # Pre-emphasis factor
# noise_threshold = 0.02  # Threshold for noise discrimination

# # Process prototypes
# for idx, file in enumerate(proto_files):
#     # Read audio
#     Fs, sig = wav.read(file)
    
#     # Normalize
#     sig = sig / np.max(np.abs(sig))
    
#     # Convert to mono if stereo
#     if len(sig.shape) > 1:
#         sig = sig[:, 0]
    
#     # Pre-emphasis
#     pre = np.roll(sig, 1) - alpha * sig
#     pre[0] = 0  # Set the first sample to 0
    
#     # Remove noise (apply threshold)
#     pre = np.where(np.abs(pre) >= 0.7, 0,pre)
    
#     # Trim active region
#     active_idx = np.where(np.abs(pre) > noise_threshold)[0]
#     if len(active_idx) > 0:
#         start_idx = active_idx[0]
#         end_idx = active_idx[-1]
#         pre_trimmed = pre[start_idx:end_idx + 1]
#     else:
#         pre_trimmed = pre
    
#     # Visualization
#     duration = len(pre)/Fs
#     time_trimmed = np.linspace(start_idx / Fs, end_idx / Fs, len(pre_trimmed))
#     time = np.linspace(0, duration, len(pre))
#     plt.figure(figsize=(10, 6))
    
#     # Original signal
#     plt.subplot(3, 1, 1)
#     plt.plot(time, sig)
#     plt.title('Señal Original')
#     plt.xlabel('Tiempo [s]')
#     plt.ylabel('Amplitud')
    
#     # Pre-emphasized signal
#     plt.subplot(3, 1, 2)
#     plt.plot(time, pre)
#     plt.title('Señal con Pre-énfasis')
#     plt.xlabel('Tiempo [s]')
#     plt.ylabel('Amplitud')
    
#     # Trimmed signal
#     plt.subplot(3, 1, 3)
#     plt.plot(time_trimmed, pre_trimmed)
#     plt.title('Señal Recortada')
#     plt.xlabel('Tiempo [s]')
#     plt.ylabel('Amplitud')
    
#     plt.tight_layout()
#     # plt.show()
    
#     # Window length and step
#     win_len = int(win_dur * Fs)
#     step = win_len
#     # Number of windows
#     n_windows = int(np.ceil((len(pre_trimmed) - win_len) / step)) + 1
#     # Pad signal
#     pad_len = n_windows * step + win_len
#     sig_pad = np.append(pre_trimmed, np.zeros(pad_len - len(pre_trimmed)))
    
#     # Energy and ZCR
#     eng = []
#     zcr = []
#     for i in range(n_windows):
#         start = i * step
#         end = start + win_len
#         win = sig_pad[start:end]
#         eng.append(np.sum(win ** 2))
#         zcr.append(np.sum(np.abs(np.diff(np.sign(win)))) / 2)
    
#     # Store features
#     proto_feats.append((idx, eng, zcr))

# # Save to CSV
# output_csv = "database.csv"

# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # Write header
#     writer.writerow(["Class", "Energy", "ZCR"])
    
#     # Write data
#     for class_idx, eng_list, zcr_list in proto_feats:
#         for eng, zcr in zip(eng_list, zcr_list):
#             writer.writerow([class_idx, eng, zcr])

# print(f"Features saved to {output_csv}")

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Prototype audio file paths
proto_files = [
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\zero_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\uno_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\dos_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\tres_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\cuatro_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\cinco_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\seis_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\siete_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\ocho_01.wav",
    r"D:\Upiita\6to\Patrones\Practicas\DTW_ard\Audios\Erik\nueve_01.wav"
]

# Store prototype features
proto_feats = []

# Window parameters
win_dur = 0.02  # 20 ms
alpha = 0.95  # Pre-emphasis factor
noise_threshold = 0.02  # Threshold for noise discrimination

# Process prototypes
for idx, file in enumerate(proto_files):
    # Read audio
    Fs, sig = wav.read(file)
    
    # Normalize
    sig = sig / np.max(np.abs(sig))
    
    # Convert to mono if stereo
    if len(sig.shape) > 1:
        sig = sig[:, 0]
    
    # Pre-emphasis
    pre = np.roll(sig, 1) - alpha * sig
    pre[0] = 0  # Set the first sample to 0
    
    # Remove noise (apply threshold)
    pre = np.where(np.abs(pre) >= 0.7, 0, pre)
    
    # Trim active region
    active_idx = np.where(np.abs(pre) > noise_threshold)[0]
    if len(active_idx) > 0:
        start_idx = active_idx[0]
        end_idx = active_idx[-1]
        pre_trimmed = pre[start_idx:end_idx + 1]
    else:
        pre_trimmed = pre
    
    # Window length and step
    win_len = int(win_dur * Fs)
    step = win_len
    # Number of windows
    n_windows = int(np.ceil((len(pre_trimmed) - win_len) / step)) + 1
    # Pad signal
    pad_len = n_windows * step + win_len
    sig_pad = np.append(pre_trimmed, np.zeros(pad_len - len(pre_trimmed)))
    
    # Energy and ZCR
    eng = []
    zcr = []
    for i in range(n_windows):
        start = i * step
        end = start + win_len
        win = sig_pad[start:end]
        eng.append(np.sum(win ** 2))
        zcr.append(np.sum(np.abs(np.diff(np.sign(win)))) / 2)
    
    # Store features
    proto_feats.append((idx, eng, zcr))

# Save to TXT
output_txt = "database.txt"

with open(output_txt, mode='w') as file:
    # Write header
    file.write("Class,Energy,ZCR\n")
    
    # Write data
    for class_idx, eng_list, zcr_list in proto_feats:
        for eng, zcr in zip(eng_list, zcr_list):
            file.write(f"{class_idx},{eng},{zcr}\n")

print(f"Features saved to {output_txt}")
