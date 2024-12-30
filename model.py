from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def cmodel(num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # 32 filtry, rozmiar 3x3
        Conv2D(64, (3, 3), activation='relu'),  # 32 filtry, rozmiar 3x3
        MaxPooling2D((2, 2)),  # Warstwa poolingowa
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),# Druga warstwa konwolucyjna
        MaxPooling2D((2, 2)),
        Flatten(),  # Spłaszczenie cech
        Dense(128, activation='relu'),  # Warstwa w pełni połączona
        Dropout(0.75),
         # Dropout, aby zapobiec przeuczeniu
        Dense(num_classes, activation='softmax')  # Warstwa wyjściowa z softmax dla klasyfikacji
    ])

    optimazer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimazer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model