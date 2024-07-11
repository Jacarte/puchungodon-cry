import numpy as np
import scipy.signal
import cv2

def mel_filter_bank(sample_rate, n_mels, n_fft):
    # Compute the mel filter bank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + sample_rate / 2 / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            filter_bank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filter_bank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filter_bank

def compute_mel_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(scipy.signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft-hop_length)[2])

    # Generate mel filter bank
    mel_filter = mel_filter_bank(sample_rate, n_mels, n_fft)

    # Apply the mel filter bank to the power spectrogram (STFT)
    mel_spectrogram = np.dot(mel_filter, stft**2)

    return mel_spectrogram

def add_noise(image, mean=0, var=0.01):
    """
    Add Gaussian noise to an image.
    
    Parameters:
    image (numpy.ndarray): Input image.
    mean (float): Mean of the Gaussian noise.
    var (float): Variance of the Gaussian noise.
    
    Returns:
    numpy.ndarray: Noisy image.
    """
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col)).reshape(row, col)
    noisy_image = image + gauss * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_noise_to_spectrogram(spectrogram, mean=0, var=0.01):
    noise = np.random.normal(mean, var ** 0.5, spectrogram.shape)
    noisy_spectrogram = spectrogram + noise
    return noisy_spectrogram