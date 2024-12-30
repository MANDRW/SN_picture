from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.profiler.profiler_client import monitor

import data
import model

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
    callbacks=[stop]
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





model.save('conv.keras')
