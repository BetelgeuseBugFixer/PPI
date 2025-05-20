uniprot_ids = []

# Extract UniProt IDs
with open("files/Pfam-A-example.fasta", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith(">"):
            uniprot_id = line.split(" ")[1]
            uniprot_ids.append(uniprot_id.split(".")[0])
    f.close()

# Remove duplicates
uniprot_ids = list(set(uniprot_ids))

# Write to CSV
with open("files/uniprot_ids.csv", "w") as f:
    for uniprot_id in uniprot_ids:
        f.write(f"{uniprot_id}\n")
    f.close()