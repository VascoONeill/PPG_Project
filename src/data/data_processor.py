import numpy as np
import pandas as pd
from scipy.interpolate import *
from window_slider import Slider
import tsfel


def sliding_window(signal, window_size, overlap):
    slider = Slider(window_size, overlap)
    slider.fit(signal)
    window_data = []
    while True:
        window_data.append(slider.slide())
        if slider.reached_end_of_list():
            break
    return window_data


def moving_maxi(window):
    """
    Auxiliary function to calculate the absolute value of a given time-window
    :param window: array of values to be processed
    :type window: numpy array
    :return: the absolute maximum of the given "window"
    :rtype: float
    """
    return np.max(abs(window), axis=0)


def moving_max(signal, window_size=512):
    """
    This function calculates the moving absolute maximum of the surrounding window of all samples.
    It does not work if the signal attributes is None or empty.
    Run band_pass_filter before running this function.
    :param window_size: window in which the maximum will be based on
    :type window_size: int
    :return: the moving maximum of the ecg signal in the "signal" attribute of the ECG variable
    :rtype:
    """
    windows = []
    for i in range(len(signal)):
        n = [int(i - window_size / 2), int(window_size / 2 + i)]
        if n[0] < window_size / 2:
            n[0] = 0
        if n[1] > np.shape(signal)[-1]:
            n[1] = np.shape(signal)[-1]
        windows.append(signal[n[0]:n[1]])
    moving_maximum = np.array([moving_maxi(window) for window in windows])
    return np.array(moving_maximum)


def segmentation(data, size_of_segment, step_size):
    # Size of each segment must have a number 2^n+1 of data points
    # List where we will store the segments of the signal
    segments = []

    # Iterate over the signal with a step size corresponding to the overlap
    for i in range(0, len(data), step_size):
        # We need to handle the case of the last segment which might not have enough values for segmentation.
        # In that case, we don't store it
        if i + size_of_segment < len(data):
            segment = data[i:i + size_of_segment]
            segments.append(segment)
    return segments


def smooth(sig, n):
    # Make an array containing only zeros and with length = length_of_signal + 2*n
    extremes_zeros = np.zeros(len(sig) + 2 * n)

    for i in range(len(sig)):
        if i < n:
            # mirror the start of signal
            extremes_zeros[i] = sig[n - i]
            # mirror the end of signal
            extremes_zeros[len(sig) + n + i] = sig[len(sig) - i - 2]
        # fill the remaining with the signal itself
        extremes_zeros[i + n] = sig[i]

    # Build the array
    smoothen_signal = np.zeros(len(sig))

    # Fill the array
    for i in range(n, len(extremes_zeros) - n):
        # Calculate the mean of the neighours - we have to look at the surrounding neighbours,
        # thus we divide the total by 2 look back and further
        mean_neighbours = np.mean(extremes_zeros[i - n // 2:i + n // 2])
        smoothen_signal[i - n] = mean_neighbours

    return smoothen_signal


def normalize(data):
    mean_value = np.mean(data)  # Calculate the mean of the signal
    abs_signal = np.abs(data)  # Calculate the absolute of the signal
    max_abs_signal = np.max(abs_signal)  # Get the maximum value of the absolute signal
    normalised_signal = (data - mean_value) / max_abs_signal

    return normalised_signal


def baseline_removal(data, baseline_factor):
    baseline_wander = smooth(data, baseline_factor)
    filtered_signal = data - baseline_wander

    return filtered_signal


def min_removal(data):
    interval_signal = data - min(data)
    return interval_signal


def quantization(signal, d):
    # Defining a given range of possible values that the signal can have.
    # d is the number of possible values.
    scaled_signal = signal * d
    rounded_signal = np.around(scaled_signal)

    quantised_signal = np.array(rounded_signal, dtype=int)

    return quantised_signal


def data_process(data, smoothing_factor, baseline_factor, fs, mov_max_factor):
    """Signal Processing for Synthesis"""
    "Normalise Data"
    normalised_signal = normalize(data)

    "Denoise Signals"
    smoothen_signal = smooth(normalised_signal, smoothing_factor)

    "Remove Baseline Wander"
    filtered_signal = baseline_removal(smoothen_signal, baseline_factor)

    "Moving Max"
    "Needs the mov_max_factor and the fs arguments"
    # mov_max = moving_max(filtered_signal, window_size=mov_max_factor)
    # mov_max_signal = filtered_signal/mov_max

    "Remove minimum"
    processed_signal = min_removal(filtered_signal)
    
    return processed_signal


def process_time(time_samples, data):
    time_samples = np.array(time_samples)
    time_dif = time_samples-time_samples[0]
    xp = np.linspace(0, time_dif[-1], len(data))

    fs = 1 / (xp[1] - xp[0])

    xp = np.linspace(0, time_dif[-1], int(len(data) * 120/fs))
    interp = make_interp_spline(time_dif, data)
    data_interp_signal = interp(xp)

    fs = 1 / (xp[1] - xp[0])

    return xp, data_interp_signal, fs


def freq_fft(data, fps, lim_inf, lim_sup):
    fft = np.fft.fft(data)

    frequences = np.fft.fftfreq(len(data), d=(1/fps))

    condition = (frequences > lim_inf) & (frequences <= lim_sup)

    indexes = np.where(condition)
    freq_targeted = frequences[indexes]
    fft_result = fft[indexes]

    fft_targeted = np.abs(fft_result)

    return freq_targeted, fft_targeted


def scipy_filter(spec_freq, spec_time, spec_int, lim_inf, lim_sup):

    reference_db = 1e-12
    spec_db = 10 * np.log10(spec_int / reference_db)

    condition = (spec_freq > lim_inf) & (spec_freq <= lim_sup)

    indexes = np.where(condition)[0]
    selected_spec_db = spec_db[indexes, :]
    time_mesh, freq_mesh = np.meshgrid(spec_time, spec_freq[indexes])
    return freq_mesh, time_mesh, selected_spec_db


def tsfel_features(feature_name, signal, window_size, overlap):
    windows = sliding_window(signal, window_size, overlap)
    feature_function = getattr(tsfel, feature_name)
    feature_values = []
    for window in windows:
        value = feature_function(window)
        feature_values.append(value)
    return feature_values


def tsfel_features_fs(feature_name, signal, window_size, overlap, fs):
    windows = sliding_window(signal, window_size, overlap)
    feature_function = getattr(tsfel, feature_name)
    feature_values = []
    for window in windows:
        value = feature_function(window, fs)
        feature_values.append(value)
    return feature_values


def save_signals_to_csv(filename, time_sample, signals):
    data = {'Time': time_sample}
    for i, signal in enumerate(signals):
        data[f'Signal_{i+1}'] = signal

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
