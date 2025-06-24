import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # Añadido OneHotEncoder
from sklearn.neural_network import MLPClassifier # Perceptrón Multicapa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Para guardar el modelo, scaler y encoder

# --- 1. Carga del Dataset ---
# Asegúrate de que 'introvert_extrovert_dataset.csv' esté en la misma carpeta que este script
try:
    df = pd.read_csv('personality_dataset.csv') # Cambiado a 'personality_dataset.csv' para mayor claridad
    print("Dataset cargado exitosamente.")
except FileNotFoundError:
    print("Error: 'introvert_extrovert_dataset.csv' no encontrado.")
    print("Asegúrate de descargar el archivo de Kaggle y colocarlo en la misma carpeta que el script.")
    exit() # Salir si el archivo no se encuentra

# --- 2. Exploración y Preprocesamiento de Datos ---

print("\n--- Vista Previa del Dataset ---")
print(df.head())

print("\n--- Información del Dataset ---")
df.info()

# --- Manejo de Valores Faltantes (Imputación con la moda para categóricas, media para numéricas) ---
# Columnas categóricas (object dtype)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Excluir la columna 'Personality' de la imputación de la moda si es la variable objetivo
if 'Personality' in categorical_cols:
    categorical_cols.remove('Personality')

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Columnas numéricas (float64 dtype)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

print("\nValores faltantes manejados: Listo")
print(df.info()) # Verificar que no hay nulos después de la imputación


# CORRECCIÓN CLAVE: El nombre de la columna objetivo es 'Personality', no 'label'
print("\n--- Conteo de Valores de la Variable Objetivo 'Personality' ---")
print(df['Personality'].value_counts())

# Identificar características (X) y la variable objetivo (y)
# Las características son todas las columnas excepto 'Personality'
X = df.drop('Personality', axis=1) # Elimina la columna 'Personality' del DataFrame X
y = df['Personality'] # Asigna la columna 'Personality' a la variable objetivo y

# --- Codificación de Variables Categóricas en X ---
# Identificar columnas categóricas en X (ahora solo 'Stage_fear' y 'Drained_after_socializing')
X_categorical_cols = X.select_dtypes(include='object').columns.tolist()

# Aplicar One-Hot Encoding a las características categóricas
# Usamos OneHotEncoder para crear nuevas columnas binarias para cada categoría
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False para array denso

# Ajustar y transformar en el conjunto de entrenamiento, solo transformar en el de prueba
X_encoded = encoder.fit_transform(X[X_categorical_cols])
# Obtener los nombres de las nuevas columnas creadas por el encoder
encoded_feature_names = encoder.get_feature_names_out(X_categorical_cols)
# Crear un DataFrame con las características codificadas
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=X.index)

# Concatenar las características numéricas y las codificadas
X_numeric = X.select_dtypes(include=np.number) # Seleccionar columnas numéricas de X
X_processed = pd.concat([X_numeric, X_encoded_df], axis=1)

print(f"\nCaracterísticas procesadas (X_processed) shape: {X_processed.shape}")
print(X_processed.head())

# Codificar la variable objetivo categórica ('introvert', 'extrovert') a numérica (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nClases originales: {label_encoder.classes_}")
print(f"Clases codificadas: {np.unique(y_encoded)}")


# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.20, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento (X_train): {X_train.shape}")
print(f"Tamaño del conjunto de prueba (X_test): {X_test.shape}")

# Escalar las características numéricas (todas las características ahora son numéricas)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nCaracterísticas escaladas: Listo")

# --- 3. Creación y Entrenamiento del Modelo MLPClassifier (Red Neuronal) ---

print("\n--- Inicializando y Entrenando el Modelo MLPClassifier ---")
model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Dos capas ocultas: 100 neuronas y 50 neuronas
    max_iter=1000,             # Aumentar iteraciones para asegurar convergencia
    activation='relu',         # Función de activación ReLU
    solver='adam',             # Optimizador Adam
    learning_rate_init=0.001,  # Tasa de aprendizaje inicial
    random_state=42,           # Semilla para reproducibilidad
    early_stopping=True,       # Habilitar parada temprana
    n_iter_no_change=20,       # Detener si no hay mejora en 20 épocas de validación
    verbose=False,              # Desactivar el progreso detallado para una salida más limpia
    validation_fraction=0.1     # Fracción de datos de entrenamiento para validación interna
)

# Entrenar el modelo con los datos escalados
model.fit(X_train_scaled, y_train)
print("Entrenamiento del modelo finalizado.")

# --- 4. Predicción y Evaluación (para llenar el Punto 4 de la Ficha Técnica) ---

# Realizar predicciones sobre el conjunto de prueba escalado
y_pred = model.predict(X_test_scaled)

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Evaluación del Modelo ---")
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Generar un reporte de clasificación detallado
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Generar la matriz de confusión
print("\n--- Matriz de Confusión ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualización de la Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# --- Guardar el modelo, scaler y encoder para uso futuro (ej. en Streamlit) ---
joblib.dump(model, 'mlp_personality_model.pkl')
joblib.dump(scaler, 'scaler_personality.pkl')
joblib.dump(label_encoder, 'label_encoder_personality.pkl')
joblib.dump(encoder, 'onehot_encoder_personality.pkl') # Guardar también el OneHotEncoder

print("\nModelo, Scaler, LabelEncoder y OneHotEncoder guardados exitosamente.")
