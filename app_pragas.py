import os
import gdown
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st

# Título do aplicativo
st.set_page_config(page_title="Diagnóstico de Pragas - Agro Essencial", page_icon="🌾", layout="centered")
st.title("🧠 Diagnóstico de Pragas - Agro Essencial")
st.markdown("Faça o upload de uma imagem da planta afetada para identificar possíveis pragas.")

# Baixando o modelo do Google Drive, se necessário
model_path = 'models/modelo_pragas_agroessencial.h5'
file_id = 'https://drive.google.com/file/d/1g-2rOrAdTP9jUr3-HHKgZIJDXsfMVpvb/view?usp=sharing'  # <--- Substitua pelo ID do seu arquivo .h5 do Google Drive

if not os.path.exists(model_path):
    st.info("Baixando modelo de IA...")
    os.makedirs('models', exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Carrega o modelo
model = load_model(model_path)

# Classes (modifique conforme as classes usadas no seu modelo)
class_names = ['Lagarta-do-cartucho', 'Percevejo-marrom', 'Mosca-branca', 'Pulgão', 'Cigarrinha-do-milho']

# Upload da imagem
uploaded_file = st.file_uploader("📤 Envie uma imagem da planta afetada", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Imagem enviada", use_container_width=True)

    # Prepara a imagem para predição
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realiza a predição
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Mostra o resultado
    st.success(f"✅ Praga identificada: **{predicted_class}** com {confidence:.2f}% de confiança.")

    