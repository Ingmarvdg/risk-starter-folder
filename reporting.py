import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

###############Load config.json and get path variables
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = config["prod_deployment_path"]
model_path = os.path.join(config['output_model_path']) 

inference_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
output_column = "exited"

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    #read the deployed model and a test dataset, calculate predictions
    model_name = "trainedmodel.pkl"

    with open(os.path.join(prod_deployment_path, model_name), 'rb') as f:
        model = pickle.load(f)

    test_x = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))[inference_columns + [output_column]]
    test_y = test_x.pop(output_column)

    pred_y = model.predict(test_x)

    matrix = metrics.confusion_matrix(test_y, pred_y)
    plot = metrics.ConfusionMatrixDisplay(matrix, display_labels=model.classes_)

    plot.plot()

    plt.savefig(os.path.join(model_path, "confusionmatrix2.png"))

if __name__ == '__main__':
    score_model()
