from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
from pathlib import Path


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config["output_model_path"])

prod_deployment_path = os.path.join(config['prod_deployment_path']) 

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for file in Path(model_path).glob("*"):
        shutil.copy(file, prod_deployment_path)

    for file in Path(dataset_csv_path).glob("*.txt"):
        shutil.copy(file, prod_deployment_path)

if __name__ == "__main__":
    store_model_into_pickle()