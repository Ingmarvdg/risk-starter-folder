from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(config["prod_deployment_path"])

inference_columns = ["lastmonth_activity","lastyear_activity","number_of_employees"]
output_column = "exited"

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(os.path.join(model_path, "trainedmodel.pkl"), 'rb') as f:
        model = pickle.load(f)

    test_x = pd.read_csv(test_data_path)[inference_columns + [output_column]]
    test_y = test_x.pop(output_column)

    predictions = model.predict(test_x)

    f1_score = metrics.f1_score(predictions, test_y)
    
    record = str(f1_score)
    with open(os.path.join(model_path, "latestscore.txt"), "a", newline="") as file:
        file.write(record + "\n")

    return record

if __name__ == "__main__":
    score_model()