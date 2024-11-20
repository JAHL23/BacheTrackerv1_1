#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Librerias
!pip install scikeras
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
from sklearn.model_selection import RepeatedKFold, cross_val_score
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers, models, regularizers, optimizers, load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input   # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# Definir los directorios de las imágenes de baches y carreteras
potholes_dir ='./image_data/pothole_data'  #Poner tu ruta de la carpeta Yonki xd
road_dir = './image_data/road_data' #Poner tu ruta de la carpeta Yonki xd

# Función para cargar imágenes desde un directorio específico y asignar etiquetas
def load_images(folder, label):
    """
    Carga imágenes desde un directorio, las redimensiona y las normaliza.

    Args:
        folder (str): Ruta del directorio donde se encuentran las imágenes.
        label (int): Etiqueta asociada a las imágenes (1 para baches, 0 para carretera).

    Returns:
        tuple: Lista de imágenes procesadas y sus etiquetas correspondientes.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Leer la imagen
        img = cv2.resize(img, (512, 512))  # Redimensionar a 512x512 píxeles
        img = img / 255.0  # Normalizar los valores de píxeles a [0, 1]
        images.append(img)
        labels.append(label)
    return images, labels

# Cargar imágenes de baches y carreteras
pothole_img, pothole_lb = load_images(potholes_dir, 1)  # Etiqueta 1 para baches
road_surface_img, road_surface_lb = load_images(road_dir, 0)  # Etiqueta 0 para carretera

# Combinar imágenes y etiquetas en un solo conjunto
X = np.array(pothole_img + road_surface_img)  # Conjunto de datos (imágenes)
y = np.array(pothole_lb + road_surface_lb)  # Etiquetas correspondientes

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Verificar las dimensiones de los conjuntos resultantes
print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}, {y_test.shape}")


# In[ ]:


# Configuración de aumentación de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rotation_range=20,         # Rotación aleatoria de hasta 20 grados
    width_shift_range=0.2,     # Desplazamiento horizontal aleatorio hasta el 20% del ancho de la imagen
    height_shift_range=0.2,    # Desplazamiento vertical aleatorio hasta el 20% del alto de la imagen
    shear_range=0.15,          # Transformación de corte (shear) hasta el 15%
    zoom_range=0.15,           # Zoom aleatorio dentro de un rango de ±15%
    horizontal_flip=True       # Inversión horizontal aleatoria
)

# Generador para los datos de validación (sin aumentación)
validation_datagen = ImageDataGenerator()  # Simplemente escala las imágenes si es necesario

# Crear el generador para el conjunto de validación
# Este generador no realiza aumentación, pero organiza los datos en lotes (batch)
validation_generator = validation_datagen.flow(
    X_test,   # Conjunto de imágenes de prueba
    y_test,   # Etiquetas de prueba
    batch_size=32  # Tamaño del lote (batch)
)

# Crear el generador para el conjunto de entrenamiento
# Este generador aplica las transformaciones definidas en `train_datagen`
train_generator = train_datagen.flow(
    X_train,  # Conjunto de imágenes de entrenamiento
    y_train,  # Etiquetas de entrenamiento
    batch_size=32  # Tamaño del lote (batch)
)


# In[ ]:


# Crear el modelo ajustado de la CNN
model = models.Sequential([
    # Capa de entrada y primera capa convolucional
    layers.Input(shape=(256, 256, 3)),  # Entrada para imágenes de 256x256 con 3 canales (RGB)
    
    # Bloque 1: Convolucional + Pooling
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),  # Regularización L2
    layers.BatchNormalization(),  # Normalización batch para estabilizar el entrenamiento
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Bloque 2: Convolucional + Pooling
    layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Bloque 3: Convolucional + Pooling
    layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Bloque 4: Convolucional + Pooling
    layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Aplanado y capas densas
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Aumentar unidades ocultas
    layers.Dropout(0.5),  # Dropout para evitar overfitting
    
    # Capa de salida
    layers.Dense(1, activation='sigmoid')  # Activación sigmoide para clasificación binaria
])

# Compilación del modelo
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),  # Reducir el learning rate para un entrenamiento más estable
    loss='binary_crossentropy',           # Pérdida para problemas de clasificación binaria
    metrics=['accuracy']                  # Métrica de precisión para evaluar el desempeño
)

# Resumen del modelo
model.summary()


# In[ ]:


# Configuración de callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',    # Monitorea la pérdida en el conjunto de validación
        patience=10,           # Detiene el entrenamiento si no mejora en 10 épocas
        restore_best_weights=True  # Restaura los pesos de la mejor época
    ),
    ModelCheckpoint(
        filepath='best_model.h5',  # Guarda el mejor modelo durante el entrenamiento
        monitor='val_loss',
        save_best_only=True        # Guarda solo si el modelo mejora
    )
]

# Entrenamiento del modelo
history = model.fit(
    train_generator,               # Generador de datos de entrenamiento
    validation_data=validation_generator,  # Generador de datos de validación
    epochs=50,                     # Número máximo de épocas
    callbacks=callbacks,           # Callbacks para detener y guardar el mejor modelo
    steps_per_epoch=len(train_generator),  # Número de lotes por época
    validation_steps=len(validation_generator),  # Lotes para validación por época
    verbose=1                      # Muestra información detallada del progreso
)

# Cargar el mejor modelo (si no se restaura automáticamente)
model = load_model('best_model.h5')

# Evaluación del modelo en el conjunto de prueba
loss, accuracy = model.evaluate(validation_generator, verbose=1)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")


# In[ ]:


# Librerías
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from tensorflow.keras import models, layers, Input

# Crear el modelo
def create_model():
    model = models.Sequential([
        Input(shape=(256, 256, 3)),  # Capa de entrada
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
    return model

# Preparar los datos (ejemplo)
# Aquí puedes reemplazar X_train y y_train por tu conjunto de datos real
X_train = np.random.rand(100, 256, 256, 3)  # 100 imágenes de ejemplo de tamaño 256x256x3
y_train = np.random.randint(2, size=100)  # 100 etiquetas binarias aleatorias (0 o 1)

# Crear el modelo usando KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Configurar la validación cruzada con 10 pliegues
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Evaluar el modelo usando validación cruzada
results = cross_val_score(model, X_train, y_train, cv=kfold)

# Mostrar los resultados
print(f'Cross-Validation Accuracy Scores: {results}')
print(f'Mean Accuracy: {results.mean()}')
print(f'Standard Deviation: {results.std()}')

