import tensorflow as tf


# Defining Callback class
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.1):
            print("\n Stopping training accuracy required reached \n")
            self.model.stop_training = True
        if(logs.get('accuracy') > 0.6):
            print("\nAccuracy is 95% reached \n")
            self.model.stop_training = True

# Retriving the data
mnist = tf.keras.datasets.mnist
(training_data, training_label), (testing_data, testing_label) = mnist.load_data()

# Scailling the data
training_data = training_data / 255.0
testing_data = testing_data / 255.0

# Creting the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
epochs = 5
callbacks = myCallback()
model.fit(training_data, training_label, epochs=5, callbacks=[callbacks])

# Evaluting the model
test_loss, test_accuracy = model.evaluate(testing_data, testing_label)
print("Test Loss: ", test_loss)
print("Test accuracy: ", test_accuracy)

# My Curiosities
import numpy as np
item_to_check = int(input("Enter the index you want to check on test data : "))
classification = list(model.predict(testing_data))
result = np.where(classification[item_to_check]
                  == max(classification[item_to_check]))

print("\nPredicted output: ", result[0][0])
print("Target output: ", testing_label[item_to_check])
