# We read each file in the data folder and process it to create a single csv file
# Each file is an audio file. Audio files are caf or wav

# We process wav files, each folder represents a class
import os
import argparse
import logging
import random
import pandas as pd
from utils import process_audio_file
import numpy as np

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
