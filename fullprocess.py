

import training
import scoring
import ingestion
import deployment
import diagnostics
import reporting
from pathlib import Path
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
with open("ingesteddata/ingestedfiles.txt", "r") as f:
    lines = f.readlines()
    existing_files = []
    for line in lines:
        existing_files.append(line.split(",")[0])

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
candidate_files = [p.name for p in Path(dataset_csv_path).glob("*.csv")]
new_files = list(set(candidate_files) - set(existing_files))

if len(new_files) > 0:
    ingestion.merge_multiple_dataframes()
else:
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open("production_deployment/latestscore.txt", "r") as f:
    early_result = float(f.readline())

if scoring.score_model() < early_result: # if current score is lower
    training.train_model()
    deployment.store_model_into_pickle()
else:
    exit()

#run diagnostics.py and reporting.py for the re-deployed model







