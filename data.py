import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

def visualize_data(images, labels, num_samples=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        a = random.randint(0, len(images) - 1)
        plt.imshow(images[a])
        plt.title(f"Etykieta: {labels[a]}")
        plt.axis('off')
    plt.show()
def datas():
    data_dir = 'CXR_png'
    picture = os.listdir(data_dir)
    img_height, img_width = 224, 224
    batch_size = 32

    data = []
    for file in picture:
        if file.endswith('.png'):
            label = int(file.split('_')[-1][0])
            file_path = os.path.join(data_dir, file)
            data.append((os.path.join(data_dir, file), label))

    images = []
    labels = []

    for file_path, label in data:
        img = load_img(file_path, target_size=(img_height, img_width))
        img_array = np.array(img) / 255.0
        images.append(img_array)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


