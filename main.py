import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import data

model = load_model('conv4.keras')

x_train, x_test, y_train, y_test = data.datas()
num_classes = 2

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

eval_results = model.evaluate(x_test, y_test, verbose=1)
print("\nWyniki ewaluacji na zbiorze testowym:")
print(f"Strata: {eval_results[0]:.4f}")
print(f"Dokładność: {eval_results[1]:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = (y_pred >= 0.5).astype(int)
y_true_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true_classes, np.argmax(y_pred, axis=1))
ConfusionMatrixDisplay(conf_matrix, display_labels=["Klasa 0", "Klasa 1"]).plot(cmap=plt.cm.Blues)
plt.title("Macierz konfuzji")
plt.show()

print("Raport klasyfikacji:")
print(classification_report(y_true_classes, np.argmax(y_pred, axis=1), target_names=["Klasa 0", "Klasa 1"]))






