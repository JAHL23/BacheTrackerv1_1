
#Libraries

import tensorflow as tf
import os
from tensorflow import keras

import numpy as np
from matplotlib import pyplot as plt

import fastapi 
import imageio 
import cv2
import imghdr



#Librerias para el modelo
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.layers import Input   # type: ignore


gpus = tf.config.list_physical_devices('GPU')
print(gpus)

#Preprocesamiento de imagenes 
potholes_dir = './pothole_image_data/Pothole_Image_Data'
road_dir ='./pothole_image_data/pp'

#etiqutas 
def load_images(folder,label):
    images=[]
    labels=[]
    for filename in os.listdir(folder):
        img_path = os.path.join(folder,filename)
        img= cv2.imread(img_path)
        img= cv2.resize(img,(256,256))
        img= img/255.0
        images.append(img)
        labels.append(label)
    return images,labels


pothole_img,pothole_lb = load_images(potholes_dir, 1) # 1 para baches

road_surface_img, road_surface_lb = load_images(road_dir, 0)  # 0 para carretera

# Combinar y dividir el dataset
X = np.array(pothole_img + road_surface_img)
y = np.array(pothole_lb + road_surface_lb)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Data augmentaton

#Flujo de imagenes 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True

)

validation_datagen = ImageDataGenerator()


# Create a generator for validation data
validation_generator = validation_datagen.flow(X_test, y_test, batch_size=32)
#Generator for train data

train_generator =train_datagen.flow(X_train,y_train,batch_size=32)



#Creamos la CNN
model = models.Sequential([
    Input(shape=(256,256,3)),#Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Salida binaria
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#Entrenar y evaluar el modelo

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(X_test) // 32
)

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')


#Función para procesar una imagen para el modelo

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Cambiar el tamaño a 256x256
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Agregar dimensión extra para el batch
    return img


from sklearn.metrics import roc_curve, auc
#Curvas ROC