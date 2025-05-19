""" This module is used to extract features from the text data"""
import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")


class FeatureExtraction:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.n_components = 3
        self.num_words = 10
        self.stop_words = stopwords.words("spanish") # Changed to a list
        self.df = None
        self.tfidf = TfidfVectorizer(min_df=1, max_df=1.0, stop_words=self.stop_words)
        self.topic_names = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def read_csv(self, path: str, file_name: str) -> pd.DataFrame:
        """Reads the CSV file and returns a DataFrame."""
        file_path = os.path.join(path, file_name)
        self.logger.info(f"este es el documento {file_path}")
        df = pd.read_csv(file_path)
        return df
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fits the TF-IDF vectorizer for a Document-Term Matrix."""
        # be sure not use NaN values
        self.df = df
        self.df = self.df.dropna(subset=["close_notes_processed"])
        self.dtm = self.tfidf.fit_transform(self.df["close_notes_processed"])
        self.logger.info(f"DATA PROCESADA EXITOSAMENTE")
    
    def topic_modeling_nmf(self):
        self.nmf = NMF(n_components=self.n_components, random_state=42)
        self.W = self.nmf.fit_transform(self.dtm)
        self.H = self.nmf.components_
        self.logger.info(f"MODELADO DE TOPOCOS TERMINADO")
    
    def assign_topics(self):
        """Asigna nombres a los topicos detectados."""
        feature_names = self.tfidf.get_feature_names_out()
        for topic_idx, topic in enumerate(self.H):
            top_words = [feature_names[i] for i in topic.argsort()[:-self.num_words - 1:-1]]
            
            if "davibox" in top_words or "legalizar" in top_words:
                topic_name = "Documentos Davibox"
            elif "reenvia" in top_words or "correo" in top_words:
                topic_name = "Reenvio de Documentos"
            else:
                topic_name = "Otro"

            self.topic_names[topic_idx] = topic_name
            self.logger.info(f"TOPICO GENERADO: ")
            self.logger.info(f"{topic_name}: {', '.join(top_words)}\n")
    
    def save_topic_mapping_to_json(self, data_path_processed, data_version):
        """Guarda los tópicos en un archivo JSON en la ruta especificada."""
        json_path = os.path.join(data_path_processed, f"topic_mapping_{data_version}.json")
        os.makedirs(data_path_processed, exist_ok=True)
        with open(json_path, "w") as file:
            json.dump(self.topic_names, file, ensure_ascii=False)
        self.logger.info(f"TOPICOS GUARDADOS EXITOSAMENTE EN {data_path_processed}")

        
    def save_df_to_csv(self, data_path_processed, data_version):
        """Guarda el dataset con los temas asignados en un CSV en la ruta especificada."""
        output_path = os.path.join(data_path_processed, f"tickets_inputs_eng_{data_version}.csv")
        self.df["tema_id"] = self.W.argmax(axis=1)
        self.df["tema_nombre"] = self.df["tema_id"].map(self.topic_names)
        self.df.to_csv(output_path, index=False)
        self.logger.info(f"DATASET GUARDADO EXITOSAMENTE EN {output_path}")

    def run(self, data_path_processed: str, data_version: int):
        df_tickets = self.read_csv(
            path=data_path_processed,
            file_name=f"tickets_{data_version}.csv",

        )
        
        self.fit(df_tickets)

        self.topic_modeling_nmf()
        self.assign_topics()
        self.save_topic_mapping_to_json(data_path_processed, data_version)
        self.save_df_to_csv(data_path_processed, data_version)
        
        self.logger.info(f"PROCESO DE EXTRACCION DE CARACTERISTICAS TERMINADO")


if __name__ == "__main__":
    feature_extractor_processor = FeatureExtraction()
    data_path_processed = "/Users/davinci/Desktop/ada_tickets_clasificator/data_project/processed_data"
    data_version = 1
    feature_extractor_processor.run(data_path_processed, data_version)