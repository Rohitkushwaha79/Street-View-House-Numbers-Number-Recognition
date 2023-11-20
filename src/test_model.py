import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model('model.hdf5')  


new_image_path = r'C:\Users\Harry\Desktop\number_model\7.jpg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (32, 32))  
new_image = np.expand_dims(new_image, axis=0)  


predictions = model.predict(new_image)


predicted_class = np.argmax(predictions)

print("Predicted Digit:", predicted_class)
