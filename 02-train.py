import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
import matplotlib.pyplot as plt
import json


ARGS = argparse.ArgumentParser()
ARGS.add_argument('--data_path', type=str, default='tmp/data.csv')
ARGS.add_argument('--seed', type=int, default=42)
ARGS.add_argument('--plot', type=bool, default=True)
ARGS.add_argument('--output', type=str, default='tmp')

LOG = logging.getLogger(__name__)
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
LOG.setLevel(logging.DEBUG)

def create_model(num_classes, size=(100,100), ):
    # Define the model
    model = Sequential()

    model.add(Reshape((*size, 1), input_shape=(size[0]*size[1], )))
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Flatten the output
    model.add(Flatten())

    # Add dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return model

def main():
    args = ARGS.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)


    data = pd.read_csv(args.data_path)

    # Set a number to each class
    data['class'] = pd.Categorical(data['class'])
    
    # Separate features and labels
    X = data.iloc[:, :-1]  # all columns except the last one
    y = data.iloc[:, -1]   # the last column (class labels)
    # Encode the class labels
    label_encoder = LabelEncoder()
    # Map the class labels to integers
    classes_map = dict(enumerate(data['class'].cat.categories))
    LOG.debug(classes_map)

    # Save the class mapping
    with open(f'{args.output}/classes.json', 'w') as f:
        json.dump(classes_map, f)

    y = label_encoder.fit_transform(y)
    # Normalize the mel spectrogram values to a range [0, 1]
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # Verify the split
    LOG.debug(f"Training set size: {X_train.shape[0]} samples")
    LOG.debug(f"Test set size: {X_test.shape[0]} samples")
    num_classes = len(np.unique(y))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    num_classes = len(np.unique(y))
    model = create_model(num_classes)
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    LOG.info(f"Test accuracy: {test_acc}")

    if args.plot:
        # Extract the training and validation accuracy values
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Extract the number of epochs
        epochs = range(1, len(train_acc) + 1)

        # Plot the training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    # Save the model
    os.makedirs(args.output, exist_ok=True)
    model.save(f'{args.output}/model.keras')
    

if __name__ == '__main__':
    main()

# After generating the csv, we augment the generated images by using some noise algorithm, e.g. Gaussian noise, etc.
