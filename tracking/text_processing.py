import os
import json
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
import logging
import warnings
import datetime
from unidecode import unidecode

warnings.filterwarnings("ignore")


class TextProcessing:
    """Clase para el preprocesamiento de texto"""

    def __init__(self, language: str):
        """Contructor para lugo crear las instancias para el preprocesamiento de texto
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        nltk.download("averaged_perceptron_tagger")
        self.language = language
        self.stop_words = set(stopwords.words(self.language))
        self.stemmer = SnowballStemmer(self.language)


    def tokenize(self, text: str):
        """Metodo para tokenizar el texto"""
        text = unidecode(text.lower())
        tokens = word_tokenize(text.lower(), language=self.language)
        return tokens

    def remove_stopwords(self, tokens: list):
        """Metodo para remover las stop words"""
        filtered_tokens = [
            word for word in tokens if word.lower() not in self.stop_words
        ]
        return filtered_tokens

    def lemmatize(self, tokens: list):
        """Metodo para lematizar el texto"""
        lemmatized_tokens = [self.stemmer.stem(word) for word in tokens]
        return lemmatized_tokens

    def pos_tagging(self, tokens: list):
        """Metodo para etiquetar el texto"""
        tagged = pos_tag(tokens)
        nouns = [word for word, pos in tagged if pos == "NN"]
        return " ".join(nouns)

    def text_preprocessing(self, column_to_process: pd.Series):
        self.logger.info(f"Inicianndo el preprocesamiento de texto")
        """Metodo para hacer el prepocesamiento de texto"""
        initial_time = datetime.datetime.now()
        tokenized_text = column_to_process.apply(self.tokenize)
        text_without_stopwords = tokenized_text.apply(self.remove_stopwords)
        text_lemma = text_without_stopwords.apply(self.lemmatize)
        pos_tagging_tokens = text_lemma.apply(self.pos_tagging)
        final_time = datetime.datetime.now()
        self.logger.info(f"Fin del preprocesamiento de texto")
        self.logger.info(f"Texo preprocesado con exito")
        self.logger.info(f"time = {final_time - initial_time}")
        
        return pos_tagging_tokens 

    def save_processed_data(self, df: pd.DataFrame, path: str, file_name: str) -> None:
        """Metodo para guardar la data procesada"""
        self.logger.info(f"Guardando la data en {path}")
        file_path = os.path.join(path, file_name)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Data guardada en {file_path}")

    def read_json(self, path: str, file_name: str):
        self.logger.info(f"Iniciando funcion read_json")
        """This method is used to read the json file"""
        file_path = os.path.join(path, file_name)
        self.logger.info(f"Esto es file path del json {file_path}")
        with open(file_path, "r") as file:
            datos = json.load(file)
        df_tickets = pd.json_normalize(datos)
        self.logger.info(f"estas son las columnas {df_tickets.columns}")
        return df_tickets

    def read_csv(self, path: str, file_name: str):
        """metodo para leer el csv"""
        self.logger.info(f"Leyendo csv")
        file_path = os.path.join(path, file_name)
        df_tickets = pd.read_csv(file_path, encoding='latin-1')
        return df_tickets

    def data_transform(self, df: pd.DataFrame):
        """metodo para transformar la data en vector"""
        
        df = df[['description', 'close_notes', 'close_code']]

        df = df.reset_index(drop=True)
        self.logger.info("Data transformada con exito")
        return df


    def run(self, file_name: str, version: int):
        """Metodo para ejecutar toda la orquestacion del pipeline"""
        name_data_input = f"{file_name}"
        PATH_DATA_RAW = "/Users/davinci/Desktop/ada_tickets_clasificator/data_project/raw_data"
        PATH_DATA_PROCESSED = "/Users/davinci/Desktop/ada_tickets_clasificator/data_project/processed_data"
        # reading JSON data
        data_tickets = self.read_csv(
            path=PATH_DATA_RAW, file_name=f"{name_data_input}.csv"
        )
        
        # data transformation
        data_tickets = self.data_transform(df=data_tickets)
        # preprocesar columna description
        processed_column = self.text_preprocessing(
            data_tickets["description"]
        )
        data_tickets["description_processed"] = processed_column
        # additional processing
        data_tickets["description_processed"] = data_tickets["description_processed"].str.replace(
            r"x+/", "", regex=True
        )
        data_tickets["description_processed"] = data_tickets["description_processed"].str.replace(
            "xxxx", ""
        )
        data_tickets = data_tickets.dropna(subset=["description_processed"])

        # preprocesar columna close_notes
        processed_column = self.text_preprocessing(
            data_tickets["close_notes"]
        )
        data_tickets["close_notes_processed"] = processed_column
        # additional processing
        data_tickets["close_notes_processed"] = data_tickets["close_notes_processed"].str.replace(
            r"x+/", "", regex=True
        )
        data_tickets["close_notes_processed"] = data_tickets["close_notes_processed"].str.replace(
            "xxxx", ""
        )
        data_tickets = data_tickets.dropna(subset=["close_notes_processed"])
        # Saving processed data
        self.save_processed_data(
            df=data_tickets,
            path=PATH_DATA_PROCESSED,
            file_name=f"{file_name}_{version}.csv",
        )

# aqui se ejeciuta la orquestacion del pipelines de preprocesamiento
if __name__ == "__main__":
    text_processing = TextProcessing(language="spanish")
    text_processing.run(file_name="tickets", version="1")