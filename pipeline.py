import mlflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
import logging
from text_processing import TextProcessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


processor = TextProcessing(language="spanish")
file_name = "test.csv"
data = processor.read_csv(
    path="/Users/davinci/Desktop/tickets_clasificator/data_project/processed_data", file_name = file_name
)

data = processor.data_transform(df=data)
close_notes_processed = processor.text_preprocessing(data["close_notes"])

test = pd.DataFrame()
test['close_notes_processed'] = close_notes_processed

#data["description_processed"] = data
#data = processor.text_preprocessing(data["close_notes"])
#data["close_notes_processed"] = data

print(test.head(5))
print(test.dtypes)

processor.save_processed_data(
    df=test,
    path="/Users/davinci/Desktop/tickets_clasificator/data_project/processed_data",
    file_name="test_processed.csv")



#data = data.dropna(subset=["description_processed"])
#data = data.dropna(subset=["close_notes_processed"])
#data = data.reset_index(drop=True)