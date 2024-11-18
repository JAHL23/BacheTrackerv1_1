# Usa una imagen base con Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requisitos e instala las dependencias
COPY app/requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt

# Copia el código de la aplicación al contenedor
COPY app/ .

# Expone el puerto que usa la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "clasi.py"]
