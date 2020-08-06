# importing the required packages
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Creating the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compliling the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Data set for trainning
x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Trainning the model
model.fit(x, y, epochs=500)

# Predicting a value
print(model.predict([10.0]))
