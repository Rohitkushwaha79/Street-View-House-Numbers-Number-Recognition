import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from data_loader import test_dataset

test_dataset= test_dataset()

model = load_model('model.hdf5')

history= model.evaluate(test_dataset)

loss = history[0]
accuracy = history[1]
top_k_accuracy = history[2]

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Top K Test accuracy: {top_k_accuracy*100:.2f}%")
