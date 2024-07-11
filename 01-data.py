# We read each file in the data folder and process it to create a single csv file
# Each file is an audio file. Audio files are caf or wav

# We process wav files, each folder represents a class
import os
import argparse
import logging
import random
import pandas as pd
import soundfile as sf
from utils import compute_mel_spectrogram, add_noise_to_spectrogram as noise
import numpy as np
import matplotlib.pyplot as plt
import cv2

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--data_path', type=str, default='data')
ARGS.add_argument('--seed', type=int, default=42)
ARGS.add_argument('--output', type=str, default="tmp")
ARGS.add_argument('--augment', type=bool, default=True)
#ARGS.add_argument('--sampling_rate', type=int, default=16000)

LOG = logging.getLogger(__name__)
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG.setLevel(logging.DEBUG)

# This function returns a ROW for the CSV file
def process_audio_file(args, file_path, filename, clazz, generate_image=True, add_noise=False):
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
        mel_spectrogram_db = noise(mel_spectrogram_db)
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




def main():
    args = ARGS.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = args.data_path
    classes = os.listdir(data_path)


    rows = []
    L = []
    CL = []

    os.makedirs(args.output, exist_ok=True)
    
    for folder in classes:
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            L.append(len(files))
            CL.append(folder)
    

    LOG.debug(f'Classes: {", ".join(CL)}')
    LOG.debug(f'Class count: {L}')

    for folder in classes:
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            
            for audio in files:
                audio_path = os.path.join(folder_path, audio)
                row = process_audio_file(args, audio_path, audio, folder)
                rows.append(row)

            if args.augment:
                to = max(L) - len(files)
                LOG.debug(f'Augmenting {folder} to {to}')
                for i in range(to):
                    audio = random.choice(files)
                    audio_path = os.path.join(folder_path, audio)
                    row = process_audio_file(args, audio_path, audio, folder, add_noise=True)
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(df)

    df.to_csv(f'{args.output}/data.csv', index=False)

            
    

if __name__ == '__main__':
    main()

# After generating the csv, we augment the generated images by using some noise algorithm, e.g. Gaussian noise, etc.
