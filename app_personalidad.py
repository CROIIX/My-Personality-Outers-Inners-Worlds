import streamlit as st
import pandas as pd
import numpy as np
import joblib # Para cargar los objetos del modelo

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="My Personality: Outers & Inners Worlds",
    page_icon="🧠", # Icono de cerebro o personalidad
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Carga de Modelos Pre-entrenados (cacheados para eficiencia) ---
@st.cache_resource
def load_artifacts():
    """Carga el modelo, scaler, label_encoder y onehot_encoder."""
    try:
        model = joblib.load('mlp_personality_model.pkl')
        scaler = joblib.load('scaler_personality.pkl')
        label_encoder = joblib.load('label_encoder_personality.pkl')
        onehot_encoder = joblib.load('onehot_encoder_personality.pkl')
        st.success("Modelo y preprocesadores cargados exitosamente.")
        return model, scaler, label_encoder, onehot_encoder
    except FileNotFoundError:
        st.error("Error: Archivos del modelo no encontrados.")
        st.write("Por favor, asegúrate de haber ejecutado el script 'modelo.py' primero para generar los archivos .pkl en la misma carpeta.")
        st.stop() # Detiene la ejecución de la app si los archivos no están
    except Exception as e:
        st.error(f"Error al cargar los archivos del modelo: {e}")
        st.stop()

model, scaler, label_encoder, onehot_encoder = load_artifacts()

# --- Definición de los Nombres de las Columnas (Orden de las características) ---
# Es CRUCIAL que el orden de las características sea el mismo que se usó para entrenar el modelo
feature_columns = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
]

# Columnas categóricas que necesitan One-Hot Encoding
categorical_features = ['Stage_fear', 'Drained_after_socializing']

# Columnas numéricas que necesitan escalado
numeric_features = [col for col in feature_columns if col not in categorical_features]

# --- Inicializar el estado de la sesión para controlar la visibilidad del botón de reinicio ---
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# --- Interfaz de Usuario del Formulario ---
st.title("Descubre Tu Personalidad: ¿Introvertido o Extrovertido?")
st.markdown("Responde las siguientes preguntas para que nuestro modelo prediga tu personalidad.")

with st.form("personality_form"):
    st.subheader("Tus Respuestas:")

    # Pregunta 1: Time_spent_Alone (Numérica)
    time_spent_alone = st.slider(
        "1. ¿Cuántas horas sueles pasar solo/a al día? (0 = ninguna, 10 = la mayor parte del día)",
        min_value=0.0, max_value=10.0, value=5.0, step=0.5
    )

    # Pregunta 2: Stage_fear (Categórica)
    stage_fear = st.radio(
        "2. ¿Sientes miedo escénico o nervios al hablar en público?",
        ('Sí', 'No')
    )

    # Pregunta 3: Social_event_attendance (Numérica)
    social_event_attendance = st.slider(
        "3. En una escala del 0 al 10, ¿con qué frecuencia asistes a eventos sociales? (0 = nunca, 10 = muy a menudo)",
        min_value=0.0, max_value=10.0, value=5.0, step=0.5
    )

    # Pregunta 4: Going_outside (Numérica)
    going_outside = st.slider(
        "4. En una escala del 0 al 10, ¿con qué frecuencia te gusta salir de casa por diversión?",
        min_value=0.0, max_value=10.0, value=5.0, step=0.5
    )

    # Pregunta 5: Drained_after_socializing (Categórica)
    drained_after_socializing = st.radio(
        "5. Después de socializar mucho, ¿te sientes agotado/a o con poca energía?",
        ('Sí', 'No')
    )

    # Pregunta 6: Friends_circle_size (Numérica)
    friends_circle_size = st.slider(
        "6. ¿Cuántos amigos cercanos tienes en tu círculo principal? (0 = pocos/ninguno, 15 = muchos)",
        min_value=0.0, max_value=15.0, value=7.0, step=1.0
    )

    # Pregunta 7: Post_frequency (Numérica)
    post_frequency = st.slider(
        "7. En una escala del 0 al 10, ¿con qué frecuencia publicas o interactúas en redes sociales?",
        min_value=0.0, max_value=10.0, value=5.0, step=0.5
    )

    # Botón de envío del formulario
    submitted = st.form_submit_button("Predecir Personalidad")

    if submitted:
        # Una vez que el formulario se envía, marcamos el estado
        st.session_state.form_submitted = True

        # --- Preprocesamiento de las entradas del usuario ---
        # Crear un DataFrame con la entrada del usuario
        user_data = pd.DataFrame([[
            time_spent_alone,
            'Yes' if stage_fear == 'Sí' else 'No', # Convertir 'Sí'/'No' a 'Yes'/'No' para el modelo
            social_event_attendance,
            going_outside,
            'Yes' if drained_after_socializing == 'Sí' else 'No', # Convertir 'Sí'/'No' a 'Yes'/'No' para el modelo
            friends_circle_size,
            post_frequency
        ]], columns=feature_columns)

        # 1. Codificar características categóricas
        user_categorical_data = user_data[categorical_features]
        user_encoded_data = onehot_encoder.transform(user_categorical_data)
        
        # Crear DataFrame con las características codificadas (con los nombres correctos)
        user_encoded_df = pd.DataFrame(
            user_encoded_data,
            columns=onehot_encoder.get_feature_names_out(categorical_features),
            index=user_data.index
        )
        
        # 2. Concatenar características numéricas y codificadas
        user_numeric_data = user_data[numeric_features]
        user_processed_data = pd.concat([user_numeric_data, user_encoded_df], axis=1)

        # 3. Escalar las características procesadas
        user_scaled_data = scaler.transform(user_processed_data)

        # --- Predicción ---
        prediction_encoded = model.predict(user_scaled_data)
        prediction_proba = model.predict_proba(user_scaled_data) # Probabilidades de cada clase

        # Decodificar la predicción a la etiqueta original (Introvert/Extrovert)
        predicted_personality = label_encoder.inverse_transform(prediction_encoded)[0]

        # Obtener la confianza (probabilidad de la clase predicha)
        confidence = np.max(prediction_proba) * 100

        # --- Mostrar Resultado ---
        st.success("Predicción Realizada:")
        if predicted_personality == "Extrovert":
            st.markdown(f"## ¡Eres **{predicted_personality}**! 🎉")
            st.write(f"Basado en tus respuestas, nuestro modelo predice que eres una persona que probablemente se energiza con las interacciones sociales y disfruta de la vida exterior. (Confianza: {confidence:.2f}%)")
        else: # Introvert
            st.markdown(f"## ¡Eres **{predicted_personality}**! 🧘‍♀️")
            st.write(f"Basado en tus respuestas, nuestro modelo predice que eres una persona que probablemente recarga energías en la tranquilidad, en actividades individuales y que prefiere entornos más íntimos. (Confianza: {confidence:.2f}%)")

        st.markdown("---")
        st.write("Recuerda que este es un modelo predictivo basado en datos y sus resultados son una estimación.")

# --- Botón para Recargar/Reiniciar el Formulario (Condicional) ---
# Este botón solo se mostrará si el formulario ya fue enviado al menos una vez
if st.session_state.form_submitted:
    st.markdown("---") # Separador visual
    if st.button("¡Repetir predicción!"):
        # Resetea el estado para que el botón de reinicio no se muestre hasta la próxima predicción
        st.session_state.form_submitted = False
        st.rerun() # Fuerza a Streamlit a reiniciar la aplicación

# Información adicional/pie de página
st.sidebar.markdown( "# **My Personality: Outers & Inners Worlds**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Desarrolladores:**\n\n - Cristian Andrés Lopera Ramírez\n\n- Esneider Franco Arias\n\n- Samuel Esteban Rojas Leyton\n\n- Sorangee Vanega Cardoza\n\n- Yeimy Sofía MAdrigal Pulido")