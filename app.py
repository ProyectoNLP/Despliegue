# Importar librerías
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random

# Título de la aplicación y mensaje de bienvenida
st.markdown("""
<style>
    body {
        background-color: #e8f0f2;
    }
    .title {
        font-size: 36px;
        color: #4a4a4a;
        font-weight: bold;
    }
    .recommendation {
        background-color: #f9f9f9;
        border: 1px solid #d1d1d1;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .emergency {
        background-color: #ffcccc;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #FF69B4;
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #FFB6C1;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<h1 class="title">Chatbot de Bienestar Emocional</h1>', unsafe_allow_html=True)

st.sidebar.image("robot.png", use_column_width=True)
st.sidebar.write("### Líneas de Emergencia")
st.sidebar.write("""- Línea 155: Orientación psicosocial y jurídica a las víctimas de violencia (24/7).
- Línea 123: Policía Nacional.
- Línea 122: Denuncias de violencia intrafamiliar, de género y violencia sexual.""")

st.markdown("---")
st.write("""¡Bienvenido a la aplicación de bienestar emocional! 😊
Aquí podrás expresar cómo te sientes y recibir recomendaciones personalizadas para mejorar tu estado de ánimo.
Ingresa tu estado emocional en el menú a continuación para obtener consejos útiles y prácticos.""")

# Diccionario de traducción de sentimientos
translation_dict = {
    'peaceful': 'tranquilo',
    'mad': 'enojado',
    'powerful': 'empoderado',
    'sad': 'triste',
    'joyful': 'alegre',
    'scared': 'asustado'
}

# Cargar el modelo y otros elementos necesarios
model = tf.keras.models.load_model('Modelo.keras')

with open('label_encoder.pkl', 'rb') as handle:
    loaded_label_encoder = pickle.load(handle)

# Cargar el tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Función para predecir el sentimiento
def predict_sentiment(model, user_input, tokenizer, loaded_label_encoder):
    try:
        # Convertir texto a secuencias numéricas usando el tokenizer
        input_sequences = tokenizer.texts_to_sequences([user_input])
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post')

        # Realizar predicción
        prediction = model.predict(input_tensor)
        predicted_class_index = np.argmax(prediction, axis=1)

        # Traducir la predicción
        predicted_label = loaded_label_encoder.inverse_transform(predicted_class_index)[0]
        translated_label = translation_dict.get(predicted_label.lower(), predicted_label)
        return translated_label

    except Exception as e:
        st.error(f"Error al predecir: {str(e)}")
        return None


# Cargar las recomendaciones desde el archivo CSV
recomendaciones_df = pd.read_csv('recomendaciones.csv', sep=';')

# Asegúrate de que el archivo tiene una columna llamada 'sentimiento' y 'recomendacion'
# Convertir el DataFrame a un diccionario
recommendations = {row['Sentiment']: row['Recomendacion'] for index, row in recomendaciones_df.iterrows()}

# Función para obtener una recomendación aleatoria
def get_recommendation(sentiment):
    if sentiment.lower() in recommendations:
        return random.choice(recommendations[sentiment.lower()].split("\n")).strip()
    else:
        return random.choice(recommendations["General"].split("\n")).strip()  # Usa recomendaciones generales si no se encuentra el sentimiento

# Entrada del usuario y ejecución de la predicción
user_input = st.text_area("¿Cómo te sientes hoy?")

if st.button("Dame una recomendación"):
    if user_input:
        sentiment_class = predict_sentiment(model, user_input, tokenizer, loaded_label_encoder)
        if sentiment_class:
            recommendation = get_recommendation(sentiment_class)
            st.write(f"Parece que hoy te sentiste: {sentiment_class.capitalize()}")
            st.write(f"Recomendación: {recommendation}")
        else:
            st.warning("No se pudo realizar la predicción.")
    else:
        st.warning("Por favor, ingresa un texto para predecir.")
