import mlflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # Necesario para cargar el vectorizador
import logging

# Configuración básica del logging para ver mensajes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configura la URI de seguimiento de MLflow (DEBE ser la misma que usaste al entrenar) ---
MLFLOW_TRACKING_PATH = "/Users/davinci/Desktop/tickets_clasificator/mlflow_tracking_data"
mlflow.set_tracking_uri(MLFLOW_TRACKING_PATH)
logging.info(f"MLflow Tracking URI configurado a: {MLFLOW_TRACKING_PATH}")

# --- 1. Define la URI del modelo que quieres cargar ---
# REEMPLAZA ESTOS VALORES CON LOS REALES DE TU CORRIDA DE MLFLOW
# Puedes encontrar el Run ID en la UI de MLflow
# El nombre del artefacto del modelo (ej. 'regresion_logistica') lo definiste en mlflow.sklearn.log_model
RUN_ID_DEL_MODELO_QUE_QUIERES_CARGAR = "4deceb9849af48749ec68b873a05b7d6" # Ejemplo: "4deceb9849af48749ec68b873a05b7d6"
NOMBRE_ARTEFACTO_DEL_MODELO = "regresion_logistica" # Reemplaza con el nombre real del artefacto (ej. 'regresion_logistica')

logged_model_uri = f'runs:/{RUN_ID_DEL_MODELO_QUE_QUIERES_CARGAR}/{NOMBRE_ARTEFACTO_DEL_MODELO}'
logging.info(f"URI del modelo a cargar: {logged_model_uri}")

# --- 2. Carga el modelo usando mlflow.pyfunc.load_model ---
try:
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
    logging.info("Modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"ERROR: No se pudo cargar el modelo desde '{logged_model_uri}'. ¿Es correcto el Run ID y el Artifact Path? Error: {e}")
    exit() # Salir si el modelo no se puede cargar

# --- 3. Cargar también el TfidfVectorizer guardado (ES CRUCIAL) ---
# El TfidfVectorizer fue logueado en la corrida PADRE de tu baseline.
# Necesitas su Run ID y el nombre del artefacto para cargarlo.
RUN_ID_CORRIDA_PADRE = "e3665fa86ddf4341b2ba1a46f1e105d4" # Este es el ID de tu "Corrida_Baseline_General"
VECTORIZER_ARTIFACT_PATH = "tfidf_vectorizer" # Este es el nombre que le diste al loguearlo

try:
    loaded_vectorizer = mlflow.sklearn.load_model(f"runs:/{RUN_ID_CORRIDA_PADRE}/{VECTORIZER_ARTIFACT_PATH}")
    logging.info("TfidfVectorizer cargado exitosamente.")
except Exception as e:
    logging.error(f"ERROR: No se pudo cargar el TfidfVectorizer desde 'runs:/{RUN_ID_CORRIDA_PADRE}/{VECTORIZER_ARTIFACT_PATH}'. Error: {e}")
    exit()

# --- 4. Carga la data real desde test.csv ---
# Asegúrate de que este archivo 'test.csv' exista y tenga la columna de texto.
RUTA_TEST_CSV = "/Users/davinci/Desktop/tickets_clasificator/data_project/processed_data/test.csv" # AJUSTA ESTA RUTA SI ES NECESARIO
COLUMNA_TEXTO_TEST = "close_notes_processed" # La columna de texto en tu test.csv

logging.info(f"Cargando data de prueba desde: {RUTA_TEST_CSV}")
try:
    df_test = pd.read_csv(RUTA_TEST_CSV)
    logging.info(f"Dataset de prueba cargado. Columnas: {df_test.columns.tolist()}")
    if COLUMNA_TEXTO_TEST not in df_test.columns:
        logging.error(f"ERROR: La columna de texto '{COLUMNA_TEXTO_TEST}' no se encuentra en el archivo {RUTA_TEST_CSV}.")
        exit()
    
    # Asegúrate de que la columna de texto sea de tipo string y maneja NaNs
    df_test[COLUMNA_TEXTO_TEST] = df_test[COLUMNA_TEXTO_TEST].astype(str).fillna('')

except FileNotFoundError:
    logging.error(f"ERROR: El archivo 'test.csv' no fue encontrado en la ruta: {RUTA_TEST_CSV}")
    exit()
except Exception as e:
    logging.error(f"ERROR al cargar el test.csv: {e}")
    exit()

# Extrae la columna de texto que se va a vectorizar
test_text_data = df_test[COLUMNA_TEXTO_TEST]
logging.info(f"Número de muestras en test.csv: {len(test_text_data)}")

# --- 5. Preprocesar la nueva data con el vectorizador cargado ---
test_vect_sparse = loaded_vectorizer.transform(test_text_data)
logging.info(f"Data de prueba vectorizada. Dimensiones: {test_vect_sparse.shape}")

# --- 6. Convertir la matriz dispersa a un Pandas DataFrame si el modelo lo requiere ---
# Aunque muchos modelos de scikit-learn aceptan matrices dispersas directamente,
# tu ejemplo sugirió usar pd.DataFrame.
# Si tu modelo funciona con matrices dispersas, puedes omitir la conversión a .toarray()
# para ahorrar memoria si tu dataset es muy grande.
test_vect_df = pd.DataFrame(test_vect_sparse.toarray())
logging.info(f"DataFrame de prueba para predicción creado. Dimensiones: {test_vect_df.shape}")

# --- 7. Realizar predicciones ---
predictions = loaded_model.predict(test_vect_df)
logging.info("Predicciones realizadas exitosamente.")

# --- 8. Opcional: Añadir las predicciones al DataFrame original o mostrar resultados ---
df_test['predicted_tema_code'] = predictions

print("\n--- Vista previa del DataFrame con predicciones ---")
print(df_test[[COLUMNA_TEXTO_TEST, 'predicted_tema_code']].head())

# Si tienes un mapeo de códigos a nombres de temas, puedes aplicarlo aquí
# Por ejemplo, si lo guardaste como un artefacto o lo puedes recrear:
# from your_module import get_label_decoder # Si lo tienes en un módulo aparte
# label_decoder = get_label_decoder()
# df_test['predicted_tema_nombre'] = df_test['predicted_tema_code'].map(label_decoder)
# print(df_test[[COLUMNA_TEXTO_TEST, 'predicted_tema_nombre']].head())

print(f"\nPredicciones completadas para {len(df_test)} muestras.")
print("El DataFrame 'df_test' ahora contiene la columna 'predicted_tema_code'.")