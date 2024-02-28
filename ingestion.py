import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframes():
    #check for datasets, compile them together, and write to an output file
    dfs = []
    file_names = os.listdir(input_folder_path)
    for file_name in file_names:
        dfs.append(pd.read_csv(file_name))

    df = pd.concat([dfs])

    df = df.drop_duplicates()

    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"))

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w", newline="") as file:
        file.write(output_folder_path)


if __name__ == '__main__':
    merge_multiple_dataframes()
