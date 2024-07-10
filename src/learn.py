import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class Mnist:

    def __init__(self, csv_path, images_folder):
        self.csv_path = csv_path
        self.images_folder = images_folder
        self.model_path = 'mnist_model.h5'
        self.df = pd.read_csv(self.csv_path)
        self.label_mapping = {str(label): i for i, label in enumerate(sorted(set(self.df['label'])))}  
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}  
        self.num_classes = len(self.label_mapping)

        # Split the data into a training and validation set
        self.train_df, self.val_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # Load and process training images
        self.X_train = np.array([self.load_and_preprocess_image(file_name) for file_name in self.train_df['file_name']])
        self.y_train = np.array([self.label_mapping[label] for label in self.train_df['label']])

        # Load and process validation images
        self.X_val = np.array([self.load_and_preprocess_image(file_name) for file_name in self.val_df['file_name']])
        self.y_val = np.array([self.label_mapping[label] for label in self.val_df['label']])

        # Build a neural network model
        self.model = keras.Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')  # Changing the number of neurons on self.num_classes
        ])

    def load_and_preprocess_image(self, image_path):
        img = cv2.imread(os.path.join(str(self.images_folder), image_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img_array = img.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        return img_array

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=100, validation_data=(self.X_val, self.y_val))

    def save_model(self):
        self.model.save(self.model_path)
        print(f'Model został zapisany do: {self.model_path}')

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print(f'Model został wczytany z: {self.model_path}')

    def predict_mnist(self):
        test_image_path = 'predict.jpg'
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        test_image = cv2.resize(test_image, (28, 28))
        test_image = test_image.astype('float32') / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=-1)
        loaded_model = tf.keras.models.load_model(self.model_path)
        predictions = loaded_model.predict(test_image)
        predicted_label = np.argmax(predictions)
        return self.inverse_label_mapping[predicted_label]  # Returning a label from the mapped value to the original label
