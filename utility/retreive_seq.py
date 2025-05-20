import requests
import pandas as pd

uniprot_ids = pd.read_csv("files/uniprot_ids.csv", header=None)  # Load UniProt IDs 

with open("files/proteins.fasta", "w") as f:
    for uniprot_id in uniprot_ids[0]:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            f.write(response.text)
        else:
            print(f"Error fetching {uniprot_id}")
    f.close()
