from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    p = request.get_json()["path"]       
    df = pd.read_csv(p)
    return model_predictions(df)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    return f"{score_model()}\n"

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats1():        
    #check means, medians, and modes for each column
    return dataframe_summary()

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats2():        
    #check timing and percent NA values
    return {
        "timing": execution_time(),
        "na_values": check_missing_data(),
    }

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
