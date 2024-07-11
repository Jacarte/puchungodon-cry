import argparse
import logging
import tensorflow as tf
import sounddevice as sd
import wavio
import pandas as pd
import json

from utils import process_audio_file

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--data_path', type=str, default='tmp/model.keras')
ARGS.add_argument('--classes', type=str, default='tmp/classes.json')
ARGS.add_argument('--output', type=str, default="tmp")
ARGS.add_argument('--sampling_rate', type=int, default=16000)

LOG = logging.getLogger(__name__)
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG.setLevel(logging.DEBUG)

def record_audio(duration, filename, sample_rate=44100):
    """
    Record audio from the microphone for a given duration and save it as a WAV file.

    Parameters:
    - duration (int): Duration of the recording in seconds.
    - filename (str): The filename for the output WAV file.
    - sample_rate (int, optional): The sample rate for the recording. Default is 44100 Hz.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished")

    # Save the recorded audio as a WAV file
    wavio.write(filename, audio_data, sample_rate, sampwidth=2)
    print(f"Saved recording to {filename}")


def main():
    args = ARGS.parse_args()

    LOG.debug("Loading model")
    model = tf.keras.models.load_model(args.data_path)
    LOG.debug("Model loaded")
    classes = json.load(open(args.classes, 'r'))
    reversed_classes = {f"{v}": k for k, v in classes.items()}
    LOG.debug(f"Classes: {classes}")

    while True:
        
        input("Press Enter to record audio")

        record_audio(duration=7, filename='infer.wav', sample_rate=args.sampling_rate)
        data = process_audio_file(args, 'infer.wav', 'infer.wav', clazz="unknown", generate_image=True)
        data =  pd.DataFrame([data]).iloc[:, :-1]  # all columns except the last one
        # Infer the class
        prediction = model.predict(data)
        class_idx = tf.argmax(prediction, axis=1).numpy()[0]

        LOG.debug(f"Predicted class index: {class_idx}")
        LOG.debug(f"Predicted class: {reversed_classes[str(class_idx)]}")




    

if __name__ == '__main__':
    main()
