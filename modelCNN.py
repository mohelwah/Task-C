'''
link to colab notebook with training job source: https://colab.research.google.com/drive/1FfIEaAyIUQX8Hdr5CMtGBrpO1fzrEiu9?usp=sharing
'''

# Python standard library
import zipfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of tensorflow errors
# matplotlib 3.5.1
import matplotlib.pyplot as plt
# tensorflow 2.8.0
import tensorflow as tf
from tensorflow import keras
# keras 2.8.0
from keras import layers
from keras.models import Sequential

#check the files data type accept by tensorflow
from pathlib import Path
import imghdr

from google.colab import drive
drive.mount('/content/gdrive')


data_dir = '/content/gdrive/MyDrive/Colab Notebooks/TaskC/dataset/dataset'
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

# training data path
trainDataPath = '/content/gdrive/MyDrive/Colab Notebooks/TaskC/dataset/dataset'
# declared constants for easy parameter value changes
batchSize = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

# load images into training dataset with 0.8 training split
trainData = tf.keras.utils.image_dataset_from_directory(
 trainDataPath,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batchSize)

# load images into validation dataset with 0.2 validation split
valData = tf.keras.utils.image_dataset_from_directory(
  trainDataPath,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=batchSize)

classNames = trainData.class_names
print(classNames)

# save training and validation dataset to cache for faster run times
AUTOTUNE = tf.data.AUTOTUNE
trainDS = trainData.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valDS = valData.cache().prefetch(buffer_size=AUTOTUNE)

numOfClasses = len(classNames)

# augment data for extra data to make models more accurate
augData = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ]
)

# sequential model created with 12 layers (+3 from augmented data)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(numOfClasses)
])

# model config with losses and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# print summary to view total parameters
model.summary()

# checkpoint callback to save highest validation accuracy model
cp = tf.keras.callbacks.ModelCheckpoint(
    "EuropeanModel.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# train the model and store it for graph creation
graph = model.fit(
  trainDS,
  validation_data=valDS,
  epochs=50,
  callbacks=[cp]
)

# store output values into appropriate variables
trainAcc = graph.history['accuracy']
valAcc = graph.history['val_accuracy']
trainLoss = graph.history['loss']
valLoss = graph.history['val_loss']
epochsRange = range(len(trainAcc))


# plot graph of the model created
plt.figure(figsize=(8, 8))
axA=plt.subplot(1, 2, 1)
axA.plot(epochsRange, trainAcc, label='Training Accuracy', linewidth=2)
axA.plot(epochsRange, valAcc, label='Validation Accuracy', color='green')
axA.legend(loc='lower right')
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title('Training and Validation Data Accuracy')

axL=plt.subplot(1, 2, 2)
axL.plot(epochsRange, trainLoss, label='Training Loss', linewidth=2)
axL.plot(epochsRange, valLoss, label='Validation Loss', color='green')
axL.legend(loc='upper right')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title('Training and Validation Data Loss')

plt.subplots_adjust(wspace=0.25)
plt.show()