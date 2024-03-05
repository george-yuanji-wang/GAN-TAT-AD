import requests
import json

r = requests.post(
    url='https://biit.cs.ut.ee/gprofiler/api/gost/profile/',
    json={
        'organism':'hsapiens',
        'query':["CASQ2", "CASQ1", "GSTO1", "DMD", "GSTM2"],
    }
    )
abc = r.json()['result']

with open(r'/Users/michelle/jupyter/ISEF/Outcome_data/gProfiler/API_access.json', 'w') as file:
    json.dump(abc, file, indent=4)