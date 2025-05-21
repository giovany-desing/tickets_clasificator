import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

# Importar infer_signature para la advertencia del modelo (si quieres añadirlo)
from mlflow.models.signature import infer_signature


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Preparación de Datos (sin cambios) ---
def preparar_datos(ruta_csv: str, x: str, y: str, test_size=0.3, random_state=42):
    logging.info(f"Cargando datos desde: {ruta_csv}")
    try:
        df = pd.read_csv(ruta_csv)
        logging.info(f"Dataset cargado exitosamente. Columnas encontradas: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"ERROR: El archivo CSV no fue encontrado en la ruta: {ruta_csv}")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"ERROR al cargar el CSV: {e}")
        return None, None, None, None, None

    if x not in df.columns:
        logging.error(f"ERROR: La columna '{x}' no se encuentra en el CSV.")
        return None, None, None, None, None
    if y not in df.columns:
        logging.error(f"ERROR: La columna '{y}' no se encuentra en el CSV.")
        return None, None, None, None, None

    logging.info(f"Columna de texto seleccionada: '{x}'")
    logging.info(f"Columna de etiqueta seleccionada: '{y}'")

    df[x] = df[x].astype(str).fillna('')
    df[y] = df[y].astype('category').cat.codes

    X_text = df[x]
    y = df[y]

    logging.info(f"Número total de muestras: {len(df)}")
    logging.info(f"Distribución de etiquetas:\n{y.value_counts(normalize=True)}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    logging.info(f"Muestras de entrenamiento: {len(X_train_text)}")
    logging.info(f"Muestras de prueba: {len(X_test_text)}")

    logging.info("Vectorizando texto con TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vect = vectorizer.fit_transform(X_train_text)
    X_test_vect = vectorizer.transform(X_test_text)
    logging.info("Vectorización completada.")
    logging.info(f"Dimensiones de X_train_vect: {X_train_vect.shape}")
    logging.info(f"Dimensiones de X_test_vect: {X_test_vect.shape}")

    return X_train_vect, X_test_vect, y_train, y_test, vectorizer

# --- 2. Función para trackear el los experimentos, evaluar y registrar el Modelo ---
def evaluar_y_registrar_modelo(model_name, model, X_train, y_train, X_test, y_test, params=None):
    """
    Entrena, evalúa y registra un modelo con MLflow como una corrida anidada.
    ASUME que ya hay una corrida padre activa.
    """
    logging.info(f"\n--- Entrenando y evaluando: {model_name} ---")

    # ¡Importante! Aquí se inicia la corrida ANIDADA
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.set_tag("mlflow.runName", model_name)
        mlflow.set_tag("dataset", "Custom_Text_CSV")
        mlflow.set_tag("model_type", "text_classifier_baseline")

        if params:
            mlflow.log_params(params)
        else:
            try:
                mlflow.log_params(model.get_params())
            except Exception as e:
                logging.warning(f"No se pudieron registrar parámetros para {model_name}: {e}")

        start_time = pd.Timestamp.now()
        model.fit(X_train, y_train)
        end_time = pd.Timestamp.now()
        training_time = (end_time - start_time).total_seconds()
        logging.info(f"Tiempo de entrenamiento: {training_time:.4f} segundos")
        mlflow.log_metric("training_time_seconds", training_time)

        y_pred = model.predict(X_test)
        # Puedes añadir la lógica para predict_proba si la necesitas, como en tu script original
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)
                # Opcional: loguear predict_proba si es útil
            except Exception as e_proba:
                logging.warning(f"No se pudo calcular predict_proba para {model_name}: {e_proba}")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision (weighted): {precision:.4f}")
        logging.info(f"Recall (weighted): {recall:.4f}")
        logging.info(f"F1-score (weighted): {f1:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)

        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_metric("f1_macro", f1_macro)

        # --- Opcional: Añadir signature e input_example para evitar la advertencia ---
        # Asegúrate de que X_test sea un numpy array para input_example
        # Si X_test_vect es una matriz dispersa, convierte una fila a array denso
        if hasattr(X_test, 'toarray'): # Si es una matriz dispersa de SciPy
            input_example_data = X_test[0].toarray()
        elif isinstance(X_test, pd.DataFrame):
            input_example_data = X_test.iloc[0].values.reshape(1, -1) # Para DataFrames
        else: # Asume numpy array u otro formato compatible
            input_example_data = X_test[0].reshape(1, -1) if X_test.ndim == 1 else X_test[0]


        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
            signature=signature,
            input_example=input_example_data
        )
        logging.info(f"Modelo {model_name} registrado en MLflow")

        run_id = run.info.run_id
        logging.info(f"MLflow Run ID para {model_name}: {run_id}")

    return model_name, accuracy, f1

# --- 3. Definición del Experimento y Ejecución del Baseline ---
def ejecutar_baseline_modelos(
    ruta_csv_param: str,
    columna_texto_param: str,
    columna_etiqueta_param: str,
    experiment_name="Baseline_Clasificacion_Texto"
):
    """Define y ejecuta el baseline para varios modelos usando un CSV de texto."""

    # 1. Configurar MLflow para usar un directorio local específico
    # Asegúrate de que esta ruta exista o pueda ser creada por MLflow.
    # Usaremos una ruta absoluta para evitar problemas de directorio de trabajo.
    mlflow_tracking_path = "/Users/davinci/Desktop/tickets_clasificator/mlflow_tracking_data" # <-- CAMBIO CLAVE AQUÍ
    mlflow.set_tracking_uri(mlflow_tracking_path)
    logging.info(f"MLflow Tracking URI configurado a: {mlflow_tracking_path}")


    # 2. Configurar el experimento (lo crea si no existe, lo selecciona si existe)
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow Experiment configurado a: {experiment_name}")


    with mlflow.start_run(run_name="Experimento") as parent_run: # <-- Corrida Padre
        logging.info(f"Iniciando corrida padre para el experimento '{experiment_name}' (Run ID: {parent_run.info.run_id})")

        # Cargar y preparar datos
        X_train_vect, X_test_vect, y_train, y_test, vectorizer = preparar_datos(
            ruta_csv_param, columna_texto_param, columna_etiqueta_param
        )

        if X_train_vect is None:
            logging.warning("No se pudieron cargar o procesar los datos. Abortando el baseline.")
            return

        # Registrar el vectorizador de texto, es crucial para la reproducibilidad
        if vectorizer:
            mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")
            logging.info("TfidfVectorizer registrado en MLflow como artefacto de la corrida padre.")


        # Definición de modelos
        models = {
            "Regresion Logistica": LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
            "SVM (Kernel Lineal)": SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
            "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5),
            "Arbol de Decision": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes Multinomial": MultinomialNB()
        }

        resultados_baseline = []

        # Bucle para evaluar cada modelo
        for nombre, modelo_instancia in models.items():
            # Cada llamada a evaluar_y_registrar_modelo ahora creará una corrida ANIDADA
            nombre_modelo_res, acc_res, f1_res = evaluar_y_registrar_modelo(
                nombre,
                modelo_instancia,
                X_train_vect, y_train, X_test_vect, y_test
            )
            resultados_baseline.append({"modelo": nombre_modelo_res, "accuracy": acc_res, "f1_weighted": f1_res})

        logging.info("\n--- Resumen del Baseline ---")
        if resultados_baseline:
            df_resultados = pd.DataFrame(resultados_baseline)
            logging.info("Resultados")
            logging.info(df_resultados.sort_values(by="f1_weighted", ascending=False))

            df_resultados.to_csv("baseline_results.csv", index=False)
            mlflow.log_artifact("baseline_results.csv")
            logging.info("Resultados del baseline guardados como artefacto.")
        else:
            logging.warning("No se generaron resultados.")

        logging.info(f"\nRevisar la UI de MLflow (ejecutar en terminal 'mlflow ui --backend-store-uri {mlflow_tracking_path}') para ver los detalles de cada corrida.")
        logging.info(f"En el experimento: {experiment_name}")

    logging.info(f"Corrida baseline completada y cerrada.")


if __name__ == "__main__":
    RUTA_A_TU_CSV = "/Users/davinci/Desktop/tickets_clasificator/data_project/processed_data/tickets_inputs_eng_1.csv"
    NOMBRE_COLUMNA_TEXTO = "close_notes_processed"
    NOMBRE_COLUMNA_ETIQUETA = "tema_nombre"

    nombre_experimento_mlflow = "Baseline_Clasificacion_tickets_ada"

    ejecutar_baseline_modelos(
        RUTA_A_TU_CSV,
        NOMBRE_COLUMNA_TEXTO,
        NOMBRE_COLUMNA_ETIQUETA,
        experiment_name=nombre_experimento_mlflow
    )