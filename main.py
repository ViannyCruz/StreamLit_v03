import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time
import requests
import zipfile
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Detector de Retinopatía Diabética",
    page_icon="👁️",
    layout="centered"
)

# Título y descripción
st.title("Detector de Retinopatía Diabética")
st.markdown("""
Esta aplicación utiliza un modelo de aprendizaje profundo (CNN) para clasificar imágenes
de fondo de retina y detectar posibles casos de retinopatía diabética.
""")

# Configuración del modelo
GITHUB_REPO = "ViannyCruz/StreamLit_v03"
MODEL_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/v1.0/best_model.h5"
MODEL_PATH = "modelo/best_model.h5"


# Función para descargar el modelo desde GitHub Releases
def descargar_modelo():
    """Descarga el modelo desde GitHub Releases si no existe localmente"""
    try:
        # Crear directorio si no existe
        os.makedirs("modelo", exist_ok=True)

        # Verificar si el modelo ya existe
        if os.path.exists(MODEL_PATH):
            return True, "Modelo encontrado localmente"

        # Descargar el modelo
        st.info("Descargando modelo desde GitHub Releases...")
        response = requests.get(MODEL_RELEASE_URL, stream=True)
        response.raise_for_status()

        # Guardar el archivo
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True, "Modelo descargado exitosamente"

    except requests.exceptions.RequestException as e:
        return False, f"Error al descargar el modelo: {str(e)}"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"


# Función para cargar el modelo local
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    try:
        # Intentar descargar el modelo si no existe
        descarga_exitosa, mensaje = descargar_modelo()

        if not descarga_exitosa:
            return None, None, mensaje

        with st.spinner("Cargando modelo..."):
            model = load_model(MODEL_PATH)
            # Obtener información sobre el tamaño de entrada esperado
            input_shape = model.input_shape
            return model, input_shape, None
    except Exception as e:
        return None, None, str(e)


# Función para preprocesar la imagen
def preprocesar_imagen(imagen, target_size):
    """Preprocesa la imagen para que sea compatible con el modelo CNN"""
    # Redimensionar la imagen al tamaño que espera el modelo
    imagen = imagen.resize(target_size)

    # Convertir a array y normalizar
    img_array = img_to_array(imagen)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización

    return img_array


# Función para realizar la predicción
def predecir_retinopatia(modelo, imagen_preprocesada):
    """Realiza la predicción usando el modelo cargado"""
    try:
        prediccion = modelo.predict(imagen_preprocesada)
        return prediccion, None
    except Exception as e:
        return None, str(e)


# Sidebar con información
with st.sidebar:
    st.title("Información")
    st.info("""
    **¿Qué es la retinopatía diabética?**
    La retinopatía diabética es una complicación de la diabetes que daña los vasos sanguíneos en la retina (la capa sensible a la luz en la parte posterior del ojo).
    """)

    st.warning("""
    **Aviso importante:**

    Esta herramienta es solo para fines educativos e informativos. No sustituye el diagnóstico profesional. Consulte siempre a un profesional de la salud.
    """)

    st.write("Desarrollado usando Streamlit y TensorFlow")

# Cargar el modelo
modelo, input_shape, error = cargar_modelo()

if modelo is not None:
    st.success("✅ Modelo cargado correctamente")
    # Determinar el tamaño de entrada esperado
    if len(input_shape) == 4:  # Normalmente (None, height, width, channels)
        target_height, target_width = input_shape[1], input_shape[2]
    else:
        # Valor predeterminado si no podemos determinar el tamaño
        target_height, target_width = 224, 224
    modelo_disponible = True
else:
    st.error(f"❌ Error al cargar el modelo: {error}")
    st.error("Detalles: Verifica la conexión a internet y la URL del modelo")
    modelo_disponible = False
    target_height, target_width = 224, 224  # Valor predeterminado

# Interfaz para subir archivos
st.subheader("Subir imagen de fondo de retina")
imagen_subida = st.file_uploader("Selecciona una imagen de fondo de retina", type=["jpg", "jpeg", "png"])

umbral_usuario = 0.05

# Si se ha subido una imagen
if imagen_subida is not None:
    # Mostrar la imagen subida
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="Imagen subida", use_container_width=True)

    # Botón para analizar la imagen
    if st.button("Analizar imagen"):
        if modelo_disponible:
            with st.spinner("Analizando imagen..."):
                # Preprocesar la imagen con el tamaño correcto
                imagen_preprocesada = preprocesar_imagen(imagen, (target_width, target_height))

                # Realizar predicción
                resultado, error_pred = predecir_retinopatia(modelo, imagen_preprocesada)

                if resultado is not None:
                    # Interpretar resultado (asumiendo que es un modelo binario)
                    probabilidad = resultado[0][0]

                    # Mostrar resultado
                    st.subheader("Resultado del análisis")

                    if probabilidad <= umbral_usuario:
                        st.error("Retinopatía diabética detectada ")
                    else:
                        st.success("No se detecta retinopatía diabética")

                    # Consejos adicionales
                    st.info(
                        "Este análisis es preliminar y no sustituye el diagnóstico médico profesional.")
                else:
                    st.error(f"Error en la predicción: {error_pred}")
                    st.info(
                        "La forma de la imagen no coincide con lo que espera el modelo. Intente con otra imagen o verifique el modelo.")
        else:
            st.error("No se puede realizar el análisis porque el modelo no está disponible.")
            st.info("Verifica la conexión a internet y la disponibilidad del modelo.")