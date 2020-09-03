# Inception dataset trained on a very big data and canbe collected as
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Importing the remaining modules
from tensorflow.keras import Model
from tensorflow.keras import layers

# We need to use a locat weights for avoiding using the pretrained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernals_notop.h5'

# Loding the pretrained model
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

# Adding the local weights
pre_trained_model.load_weights(local_weights_file)

# Setting up all the layers to non trainable
for layer in pre_trained_model:
    layer.trainable = False

# We can look up summary of the pretrained model for reference
# pre_trained_model.summary()

# Taking the last layer developing our - add on model
last_layer = pre_trained_model.get_layer('mixed7')

# Taking their output
last_output = last_layer.output

# Ok that's it its time to create our add on model
# We are importing RMSprop for optimizer
from tensorflow.keras.optimizers import RMSprop

# Our add on model
# Adding Flatten so output layer become one dimension
x = layers.Flatten()(last_output)

# Adding Dense layer
x = layers.Dense(1024, activation='relu')(x)

# Now we need a drop out of 20% to avoid overfitting so
x = layers.Dropout(0.2)(x)

# Our output layer for binary classification
x = layers.Dense(1, activation='sigmoid')(x)


# Wrapping up the entire model by adding our add-on model to the
# pre-trained model
model = Model(pre_trained_model.input, x)

# Finally its time to compile the model
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Now we need our data
!wget - -no - check - certificate \
        https:
    //storage.googleapis.com / mledu - datasets / cats_and_dogs_filtered.zip \
        - O / tmp / cats_and_dogs_filtered.zip

# Importing ImageDataGenerator for train and vaildation data-gen
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Now we need the os and zipfile modules
import os
import zipfile

# Assigning path variable
local_zip = '//tmp/cats_and_dogs_filtered.zip'

# Unzipping
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Our directories
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'vaildation')

# Data generator for training data, adding image augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# Data generator for validation data
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Train generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Validation generator
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=20,
                                                              class_mode='binary',
                                                              target_size=(150, 150))


# Time to fit the data
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_steps=50,
    verbose=2)


# Checking the accuracy by plotting the data using matplotlib
import matplotlib as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Trining accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title("Traing and validation accuracy")
plt.legend(loc=0)
plt.figure()

plt.show()
