from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

dataset_name = "finaldata.csv"
model_name = "trainedmodel.pkl"
train_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
output_column = "exited"

#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    train_df_path = os.path.join(dataset_csv_path, dataset_name)
    train_x = pd.read_csv(train_df_path)[train_columns+[output_column]]
    train_y = train_x.pop(output_column)

    model.fit(train_x, train_y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_full_path = os.path.join(model_path, model_name)

    with open(model_full_path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()