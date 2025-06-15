from collections import defaultdict, Counter
from dataset.analyse_pfams import get_data

INCLUDE_OTHER_PFAMS = False
MAX_PROTEIN_LENGTH = 1000
MIN_NUMBER_OF_PFAM_OCCURRENCES = 10
MIN_CLUSTER_SIZE = 2


def get_clusters(cluster_tsv_path):
    clusters = defaultdict(set)
    with open(cluster_tsv_path, "r") as cluster_tsv:
        for line in cluster_tsv:
            if not line:
                continue
            fields = line.split("\t")
            clusters[fields[0]].add(fields[1].strip())

    return clusters


def df_to_fasta(df, output_path):
    with open(output_path, "w") as fasta:
        for id, row in df.iterrows():
            fasta.write(f">{id}\n{row['sequence']}\n")


def check_for_max_protein_length(row):
    return len(row["sequence"]) <= MAX_PROTEIN_LENGTH


def check_if_pfams_are_in_subset(row, subset_pfams):
    pfams_in_this_protein = set(row["pfam_tensor"])
    pfams_in_this_protein.discard(None)
    if INCLUDE_OTHER_PFAMS:
        return bool(pfams_in_this_protein & subset_pfams)
    else:
        return pfams_in_this_protein.issubset(subset_pfams)


def filter_dataset_row(row, subset_pfams):
    return check_for_max_protein_length(row) and check_if_pfams_are_in_subset(row, subset_pfams)


def build_subset(data, subset_pfams):
    mask = data.apply(
        lambda row: filter_dataset_row(row, subset_pfams),
        axis=1
    )
    return data[mask]


def get_represented_pfams(subset,subset_pfams):
    counter = Counter()
    for pfam_list in subset['pfam_tensor']:
        unique_pfams = set(pfam_list) & subset_pfams
        counter.update(unique_pfams)
    return set([k for k, v in counter.items() if v >= MIN_NUMBER_OF_PFAM_OCCURRENCES])


def write_set_to_list(set_to_write, output_path):
    with open(output_path, "w") as out_file:
        for item in set_to_write:
            out_file.write(f"{item}\n")


def main():
    cluster = get_clusters("dataset/cluster/mmseqs_res_cluster.tsv")
    subset_pfams = set()
    for pfams in cluster.values():
        if len(pfams) >= MIN_CLUSTER_SIZE:
            subset_pfams |= pfams

    data = get_data()
    # filter by max protein length and cluster size
    subset = build_subset(data, subset_pfams)

    while True:
        new_pfams = get_represented_pfams(subset, subset_pfams)
        if len(new_pfams)==len(subset_pfams):
            break
        subset_pfams = new_pfams
        subset = build_subset(subset, subset_pfams)

    print(f"{data.shape}->{subset.shape}")
    print(len(subset_pfams))

    subset.to_parquet("dataset/subsets/complicated_subset/data.parquet")
    df_to_fasta(subset, "dataset/subsets/complicated_subset/data.fasta")
    write_set_to_list(subset_pfams, "dataset/subsets/complicated_subset/pfam_ids.txt")


if __name__ == '__main__':
    main()
