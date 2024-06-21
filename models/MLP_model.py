import cv2
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os
from sklearn.metrics import ConfusionMatrixDisplay

digits = load_digits()

X, y = digits.data, digits.target

X = X / np.max(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(512,), max_iter=50, solver='adam', verbose=True, learning_rate_init=0.05)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
plt.plot(model.loss_curve_)
plt.show()
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
cm.plot()
plt.show()
