import tensorflow as tf
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

train_dataset = tf.data.Dataset.load("train_dataset")
test_dataset = tf.data.Dataset.load("test_dataset")

model = tf.keras.Sequential([tf.keras.layers.LSTM(64, input_shape=(128, 9)), # (64,)
    tf.keras.layers.Dense(64, activation="relu"), # (64,)
    tf.keras.layers.Dense(6, activation="softmax") # (6,)
]) 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_dataset, epochs=30, validation_data=test_dataset)

y_true = []
y_pred = []
for X_batch, y_batch in test_dataset:
    predictions = model.predict(X_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))
y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)*100
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)