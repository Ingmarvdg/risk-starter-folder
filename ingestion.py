import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

output_location = os.path.join(output_folder_path, "finaldata.csv")

#############Function for data ingestion
def merge_multiple_dataframes():
    #check for datasets, compile them together, and write to an output file
    dfs = []
    records = []
    input_file_paths = Path(input_folder_path).glob("*.csv")
    for file_path in input_file_paths:
        logging.info(f"Found file at {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Dataframe loaded with columns {list(df.columns)} and length {len(df)}")
        dfs.append(df)
        records.append(",".join([file_path.name, str(datetime.now()), output_location]))

    df = pd.concat(dfs)

    df = df.drop_duplicates()

    df.to_csv(output_location)
    logging.info(f"Wrote dataframe with length {len(df)} to {output_location}")

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "a", newline="") as file:
        for record in records:
            file.write(record + "\n")


if __name__ == '__main__':
    merge_multiple_dataframes()
