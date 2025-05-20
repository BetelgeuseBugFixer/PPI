import numpy as np
import pandas as pd

PROTEIN_ALPHABET = {
    '-': 0,
    'A': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'V': 18,
    'W': 19,
    'Y': 20,
}

# extracts all unique pfams from pfam_tensor column
def extract_unique_pfams(pfam_tensors: pd.Series) -> list:
    all_pfams = set()
    for row in pfam_tensors:
        uniques = set(row)
        all_pfams.update(uniques)
    if None in all_pfams:
        all_pfams.remove(None)
    return list(all_pfams)

# converts pfam_tensors to one-hot encoding
def one_hot_encode_pfams(pfam_tensors: pd.Series, all_pfams: list) -> list:
    one_hot = []
    for tensor in pfam_tensors:
        p = np.zeros((len(tensor),len(all_pfams)))
        for i in range(len(tensor)):
            if tensor[i] in all_pfams:
                p[i][all_pfams.index(tensor[i])] = 1
        one_hot.append(p)
    return one_hot


def write_fasta_file(dataframe: pd.DataFrame, fasta_file_path: str) -> None:
    with open(fasta_file_path, 'w') as f:
        for _, row in dataframe.iterrows():
            f.write(f'>{row["protein_id"]}\n')
            f.write(f'{row["sequence"]}\n')