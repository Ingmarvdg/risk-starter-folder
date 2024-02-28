import requests
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"



#Call each API endpoint and store the responses
response1 = requests.post(os.path.join(URL, "prediction"), json={"path": "testdata/testdata.csv"}).text
response2 = requests.get(os.path.join(URL, "scoring")).text
response3 = requests.get(os.path.join(URL, "summarystats")).text
response4 = requests.get(os.path.join(URL, "diagnostics")).text

#combine all API responses
responses = response1 + response2 + response3 + response4

#write the responses to your workspace
with open("apireturns.txt", "w") as f:
    f.write(responses)

