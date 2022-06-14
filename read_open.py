import mne
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

def read_file(eeg_fname, time_fname, filter=(0.1, 100), r=False):
    ch_names = ['FT9', 'TP9', 'FT10', 'TP10', 'T7', 'T8']
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'])
    eeg_file = open(eeg_fname, "r")
    time_file = open(time_fname, "r")
    target = ["3", "13", "23", "33", "43"]
    time_stamp = 0
    x = []
    y = []
    b, a = signal.iirnotch(60, 20, fs=250)
    # b1, a1 = signal.iirnotch(50, 30, fs=250)
    t_stamp = []
    raw = []
    for line in eeg_file.readlines():
        if len(line.split()) > 1:
            # c = []
            # c.append(line.split()[1])
            # c.extend(line.split()[3:])
            c = line.split()[1:]
            raw.append([float(i) for i in c])
        else:
            t_stamp.append(float(line))
    # print(np.shape(raw))
    raw = signal.lfilter(b, a, raw, axis=0)
    # raw = signal.lfilter(b, a, raw, axis=0)
    # raw = signal.lfilter(b1, a1, raw, axis=0)
    raw = butter_bandpass_filter(raw, filter[0], filter[1], 250)
    # print(np.shape(raw))
    if r:
        return mne.io.RawArray(np.transpose(raw), info)
    c = 0
    for line in time_file.readlines():
        line = line.rstrip('\n')
        time = line.split(" ,")[0]
        label = line.split(" ,")[1]

        if label in target:

            while t_stamp[c] < float(time):
                c += 1
            data = []
            for j in range(500):
                data.append(raw[c-25])
                c += 1
            x.append(np.transpose(data))
            # if target.index(label) > 1:
            #     y.append(1)
            # else:
            #     y.append(0)

            # if label == "3":
            #     y.append(0)
            # else:
            #     y.append(1)
            y.append(target.index(label))
    # print(np.shape(x))
    events = [range(len(y)), [0]*len(y), y]
    events = np.asarray(events)
    events = events.T
    # print(np.shape(events))


    epoch = mne.EpochsArray(x, info=info, events=events, event_id={target[0]: 0, target[1]: 1, target[2]: 2, target[3]: 3, target[4]: 4})
    epoch = epoch.apply_baseline(baseline=(0, 0.1))
    scale = mne.decoding.Scaler(epoch.info)
    x = scale.fit_transform(epoch.get_data())
    # print(x[0][0])
    epoch = mne.EpochsArray(x, info=info, events=events, event_id={target[0]: 0, target[1]: 1, target[2]: 2, target[3]: 3, target[4]: 4})
    montage = mne.channels.make_standard_montage('standard_1005')
    epoch.set_montage(montage)
    # epochs.filter(60, 50)
    return epoch

# rms, zero crossing rate, moving window average, kurtosis, power spectral entropy
#n magnitude of FFT, discrete time wavelet based spectral entropy (db4)
# power spectral entrophy (delta, theta, alpha and beta), hurst exponent, petrosian fractal dimension
def compute_spectral(signal, sampling_rate, bands=None):
    psd = np.abs(np.fft.rfft(signal)) ** 2
    psd /= np.sum(psd)  # psd as a pdf (normalised to one)

    if bands is None:
        power_per_band = psd[psd > 0]
    else:
        freqs = np.fft.rfftfreq(signal.size, 1 / float(sampling_rate))
        bands = np.asarray(bands)

        freq_limits_low = np.concatenate([[0.0], bands])
        freq_limits_up = np.concatenate([bands, [np.Inf]])

        power_per_band = [np.sum(psd[np.bitwise_and(freqs >= low, freqs < up)])
                          for low, up in zip(freq_limits_low, freq_limits_up)]

        power_per_band = np.array(power_per_band)[np.array(power_per_band) > 0]

    spectral = - np.sum(power_per_band * np.log2(power_per_band))
    return spectral

import librosa
import scipy
import math
def extract_feature_set_one(epoch, shape='ect'):
    feature_set = []
    sr = 250

    if np.shape(epoch)[2] > 600:
        sr = 500
    for i in range(np.shape(epoch)[0]):
        features = []
        for j in range(int(np.shape(epoch)[2] / (sr/50))):
            ch = []
            for k in range(np.shape(epoch)[1]):
                e = epoch[i][k][j*int(sr/50):(j+1)*int(sr/50)]
            # zero_crossing = librosa.feature.zero_crossing_rate(epoch[i][k], frame_length=len(epoch[i][k]))
            #     print(e)
                # print('zc')
                zero_crosses = np.nonzero(np.diff(e > 0))[0]
                zero_crossing = zero_crosses.size
                # print('kt')
                kurtosis = scipy.stats.kurtosis(e)
                if math.isnan(kurtosis):
                    kurtosis = 0
                # k1 = scipy.stats.kurtosis(epoch[0][1])
                # print(np.shape(kurtosis))
                # print(k1)
                # print(kurtosis[0][1])
                # print('spectral')
                spectral = compute_spectral(e, sr)
                # print('rms')
                rms = np.sqrt(np.mean(np.square(e)))
                mean = np.mean(e)
                feature = [zero_crossing, kurtosis, spectral, rms, mean]

                # print("epoch")
                # print(e)
                # print("feature")
                # print(feature)
                ch.extend(feature)
            features.append(ch)
        # print(np.shape(features))
        if shape == 'ect':
            feature_set.append(np.transpose(features))
        else:
            feature_set.append(features)
    # print(np.shape(feature_set))
    return np.asarray(feature_set)


def read_band_features(eeg_fname, time_fname, filterband = [(0.1, 4),(4,8),(8,15),(15,32),(32,None)]):
    ch_names = ['FT9', 'TP9', 'FT10', 'TP10', 'T7', 'T8']
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'])
    eeg_file = open(eeg_fname, "r")
    time_file = open(time_fname, "r")
    target = ["3", "13", "23", "33", "43"]
    time_stamp = 0
    x = []
    y = []
    b, a = signal.iirnotch(60, 30, fs=250)
    # b1, a1 = signal.iirnotch(50, 30, fs=250)
    t_stamp = []
    raw = []
    for line in eeg_file.readlines():
        if len(line.split()) > 1:
            c = line.split()[1:]
            raw.append([float(i) for i in c])
        else:
            t_stamp.append(float(line))
    # print(np.shape(raw))
    raw = signal.lfilter(b, a, raw, axis=0)
    # raw = signal.lfilter(b1, a1, raw, axis=0)
    # raw = butter_bandpass_filter(raw, filter[0], filter[1], 250)
    fr = []
    # r = mne.io.RawArray(np.transpose(raw), info)
    for i in len(filterband):
        fr.append(butter_bandpass_filter(raw, filterband[i][0], filterband[i][1], 250))

    c = 0
    e = [[]]*len(fr)
    for line in time_file.readlines():
        line = line.rstrip('\n')
        time = line.split(" ,")[0]
        label = line.split(" ,")[1]

        if label in target:

            while t_stamp[c] < float(time):
                c += 1
            for ch in len(fr):
                data = []
                for h in range(500):
                    data.append(fr[h][c])
                    c += 1
                e[ch].append(np.transpose(data))
            # x.append(np.transpose(data))
            y.append(target.index(label))
    events = [range(len(y)), [0]*len(y), y]
    events = np.asarray(events)
    events = events.T
    epoch_list = []
    for ch in len(fr):
        epoch = mne.EpochsArray(e[ch], info=info, events=events, event_id={target[0]: 0, target[1]: 1, target[2]: 2, target[3]: 3, target[4]: 4})
    # epoch = mne.EpochsArray(x, info=info, events=events, event_id={target[0]: 0, target[1]: 1, target[2]: 2, target[3]: 3, target[4]: 4})
        scale = mne.decoding.Scaler(epoch.info)
        x = scale.fit_transform(epoch.get_data())
        epoch_list.append(mne.EpochsArray(x, info=info, events=events, event_id={target[0]: 0, target[1]: 1, target[2]: 2, target[3]: 3, target[4]: 4}))
    return epoch_list

# print(complexity["Entropy_Spectral"])
# epoch = read_file("Ear/0515_SH01/0515_PRESTIM_eeg_time_50t5c2s32ch_SH01_1.txt", "Ear/0515_SH01/0515_PRESTIM_aligned_speech_10t5c2s32ch_SH01_1.txt")
# f = extract_feature_set_one(epoch.get_data())
# print(np.shape(f))