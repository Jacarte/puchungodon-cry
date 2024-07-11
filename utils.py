import numpy as np
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import logging
import random
import soundfile as sf


LOG = logging.getLogger(__name__)
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG.setLevel(logging.DEBUG)


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


# This function returns a ROW for the CSV file
def process_audio_file(args, file_path, filename, clazz, generate_image=False, add_noise=False):
    LOG.debug(f'Processing {file_path}')
    # We take a fast fourier transform (FFT) of a 2048 sample window, slide it by 512 samples and repeat the process of the 7-sec clip. The resulting representation can be shown as a 2D image and is called a Short-Time Fourier Transform (STFT)
    audio, samplerate = sf.read(file_path)

    # Compute the mel spectrogram
    mel_spectrogram = compute_mel_spectrogram(audio, samplerate)

    # Convert the mel spectrogram to dB scale
    mel_spectrogram_db = 10 * np.log10(mel_spectrogram + 1e-6)

    # Print the number of pixels in the image
    LOG.debug(f'Image shape: {mel_spectrogram_db.shape}')

    # Normalize image to 100x100 by reducing the image, not the data, use opencv
    # Resize the image to 100x100
    mel_spectrogram_db = cv2.resize(mel_spectrogram_db, (100, 100), interpolation=cv2.INTER_CUBIC)
    
    prepend = '0'
    if add_noise:
        mel_spectrogram_db = add_noise_to_spectrogram(mel_spectrogram_db)
        prepend = random.randint(1, 1000)


    if generate_image:
        # Plot and save the mel spectrogram as an image
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.savefig(f'{args.output}/{clazz}.{prepend}.{filename}.spectrogram.png')
        # plt.show()

    # Create a row for the CSV file
    row = {
        f'{x}_{y}': mel_spectrogram_db[x, y] for x in range(100) for y in range(100)
    }
    row['class'] = clazz

    return row
