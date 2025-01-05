from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.profiler.profiler_client import monitor

import data
import model
from tensorflow.keras.callbacks import Callback


class WeightHistory(Callback):
    def __init__(self):
        super().__init__()
        self.input_weights = []
        self.output_weights = []

    def on_epoch_end(self, epoch, logs=None):
        # Pobieranie wag warstwy wejściowej i wyjściowej
        input_weights = np.mean(self.model.layers[0].get_weights()[0])
        output_weights = np.mean(self.model.layers[-1].get_weights()[0])

        self.input_weights.append(input_weights)
        self.output_weights.append(output_weights)

weight_history = WeightHistory()

x_train, x_test, y_train, y_test = data.datas()
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
model = model.cmodel(num_classes)

stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[stop,weight_history]
)


# Wykres strat (loss)
plt.plot(history.history['loss'], label='Strata - zbiór treningowy')
plt.plot(history.history['val_loss'], label='Strata - zbiór walidacyjny')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.title('Strata podczas trenowania')
plt.legend()
plt.show()

# Wykres dokładności (accuracy)
plt.plot(history.history['accuracy'], label='Dokładność - zbiór treningowy')
plt.plot(history.history['val_accuracy'], label='Dokładność - zbiór walidacyjny')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.title('Dokładność podczas trenowania')
plt.legend()
plt.show()

# Wykres błędów klasyfikacji
train_errors = [1 - acc for acc in history.history['accuracy']]
val_errors = [1 - acc for acc in history.history['val_accuracy']]

plt.plot(train_errors, label='Błąd klasyfikacji - zbiór treningowy')
plt.plot(val_errors, label='Błąd klasyfikacji - zbiór walidacyjny')
plt.xlabel('Epoki')
plt.ylabel('Błąd klasyfikacji')
plt.title('Błąd klasyfikacji podczas trenowania')
plt.legend()
plt.show()

# Predykcje na zbiorze treningowym i testowym
y_train_pred = (model.predict(x_train) >= 0.5).astype(int)
y_test_pred = (model.predict(x_test) >= 0.5).astype(int)

# Obliczenie dokładności i błędów klasyfikacji
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_error = 1 - train_accuracy
test_error = 1 - test_accuracy

print(f"Dokładność na zbiorze treningowym: {train_accuracy:.2f}, Błąd: {train_error:.2f}")
print(f"Dokładność na zbiorze testowym: {test_accuracy:.2f}, Błąd: {test_error:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(weight_history.input_weights, label="Wagi warstwy wejściowej")
plt.plot(weight_history.output_weights, label="Wagi warstwy wyjściowej")
plt.xlabel("Epoki")
plt.ylabel("Średnia wartość wag")
plt.title("Średnia wartość wag w funkcji epok")
plt.legend()
plt.show()



model.save('conv4.keras')
