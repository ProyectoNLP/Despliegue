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

# Música de fondo con control de volumen
st.markdown("### Música 🎶")
audio_file = open('musica.mp3', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/mp3', start_time=0)

st.sidebar.image("robot.png", use_column_width=True)
st.sidebar.header("Asistente Virtual")
st.sidebar.write("Si mi recomendación no es suficiente, por favor consulta las siguientes líneas de emergencia:")
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

# Diccionario de emojis por sentimiento
emoji_dict = {
    'tranquilo': '😌',
    'enojado': '😠',
    'empoderado': '💪',
    'triste': '😢',
    'alegre': '😊',
    'asustado': '😨',
    'General': '😓'  # Carita con una lágrima en la frente
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

# Convertir el DataFrame a un diccionario donde cada 'Sentimiento' tiene una lista de recomendaciones
recommendations = recomendaciones_df.groupby('Sentimiento')['Recomendacion'].apply(list).to_dict()

# Función para obtener una recomendación aleatoria
def get_recommendation(sentiment):
    sentiment = sentiment.lower()
    if sentiment in recommendations:
        return random.choice(recommendations[sentiment])  # Escoge aleatoriamente de la lista
    else:
        # Asegúrate de que haya recomendaciones generales si no se encuentra el sentimiento
        return random.choice(recommendations.get("General", ["No hay recomendaciones disponibles."])) # Usa recomendaciones generales si no se encuentra el sentimiento

# Entrada del usuario y ejecución de la predicción
user_input = st.text_area("¿Cómo te sientes hoy?")

if st.button("Dame una recomendación"):
    if user_input:
        sentiment_class = predict_sentiment(model, user_input, tokenizer, loaded_label_encoder)
        if sentiment_class:
            recommendation = get_recommendation(sentiment_class)
            emoji = emoji_dict.get(sentiment_class, '')
            st.write(f"Parece que hoy te sentiste {sentiment_class} {emoji}")
            st.write(recommendation) 
            
            # Previsualización del podcast de Spotify
            st.markdown("Te dejo una guía más profunda:")
            if sentiment_class == 'triste':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/4dBS0Murh9gq3bfdxYk586' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            elif sentiment_class == 'enojado':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/18HT8O4q8xlnYeZT5ZsG1u' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            elif sentiment_class == 'alegre':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/3xNzy6pX69kxzbXyBxdKhH' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            elif sentiment_class == 'tranquilo':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/0fVpgf337RUpiznN9OWC41' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            elif sentiment_class == 'asustado':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/072dbrBYhLYGSf2dBN1ri0' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            elif sentiment_class == 'empoderado':
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/32qnzifpWXdB9nxagTwDGN' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)
            else:  # Para el sentimiento "General"
                st.markdown("<iframe src='https://open.spotify.com/embed/episode/2s9xotmpUgEbrR7mmGEC9m' width='300' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>", unsafe_allow_html=True)

        else:
            st.warning("No se pudo realizar la predicción.")
    else:
        st.warning("Por favor, ingresa un texto para predecir.")
