
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

from ingestion import merge_multiple_dataframes
from training import train_model

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = config["prod_deployment_path"]

inference_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]

##################Function to get model predictions
def model_predictions(df: pd.DataFrame):
    #read the deployed model and a test dataset, calculate predictions
    model_name = "trainedmodel.pkl"

    with open(os.path.join(prod_deployment_path, model_name), 'rb') as f:
        model = pickle.load(f)

    test_x = df[inference_columns]

    return model.predict(test_x).tolist()

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    return df.describe().to_dict()

def check_missing_data():
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    return df.isna().sum().to_dict()

##################Function to get timings
def execution_time():
    result = [None, None]
    checkpoint = timeit.default_timer()

    merge_multiple_dataframes()

    ingest_time = timeit.default_timer() - checkpoint

    checkpoint = timeit.default_timer()

    train_model()

    train_time = timeit.default_timer() - checkpoint

    return [ingest_time, train_time]

##################Function to check dependencies
def outdated_packages_list():
    return subprocess.run(["pip", "list", "--outdated"], capture_output=True, text=True).stdout


if __name__ == '__main__':
    dataframe_summary()
    execution_time()
    outdated_packages_list()







    
