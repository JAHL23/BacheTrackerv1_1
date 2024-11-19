from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os

app = FastAPI()

# Cargar el modelo una vez al iniciar la aplicación
model = load_model('C:/Users/herre/Escritorio/semestre_7/Proyecto 1/RepoBache/BacheTrackerv1_1/app/model/modelapi_shitty.h5')

# Función para procesar y predecir si hay un bache en la imagen
def bache(image_data):
    try:
        # Preprocesar la imagen para que pueda ser entrada al modelo
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((256, 256))  # Cambiar el tamaño a 256x256
        img = image.img_to_array(img)  # Convertir a array
        img = img / 255.0  # Normalizar
        img = np.expand_dims(img, axis=0)  # Agregar dimensión extra para el batch

        # Realizar la predicción
        prediccion = model.predict(img)  # Predecir
        predicted_class = int(prediccion[0][0] > 0.5)  # Asumir que es una clasificación binaria

        return predicted_class
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# Endpoint para subir una imagen y obtener la predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer la imagen cargada
    image_data = await file.read()
    
    # Realizar la predicción de la letra
    bache_predicho = bache(image_data)
    
    # Retornar la predicción como respuesta JSON
    return {"Bache": bache_predicho}

def limpiar_salida_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')