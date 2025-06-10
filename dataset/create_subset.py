from collections import defaultdict
from dataset.analyse_pfams import get_data

INCLUDE_OTHER_PFAMS = False


def get_clusters(cluster_tsv_path):
    clusters = defaultdict(set)
    with open(cluster_tsv_path, "r") as cluster_tsv:
        for line in cluster_tsv:
            if line == "":
                continue
            fields = line.split("\t")
            clusters[fields[0]].add(fields[1].strip())

    return clusters


def df_to_fasta(df, output_path):
    with open(output_path,"w") as fasta:
        for id, row in df.iterrows():
            fasta.write(f">{id}\n{row['sequence']}\n")


def check_if_pfams_are_in_subset(row, subset_pfams):
    pfams_in_this_protein = set(row["pfam_tensor"])
    pfams_in_this_protein.discard(None)
    if INCLUDE_OTHER_PFAMS:
        return bool(pfams_in_this_protein & subset_pfams)
    else:
        return pfams_in_this_protein.issubset(subset_pfams)


def main():
    cluster = get_clusters("dataset/cluster/mmseqs_res_cluster.tsv")
    subset_pfams = set()
    for pfams in cluster.values():
        if len(pfams) > 1:
            subset_pfams |= pfams

    data = get_data()
    mask = data.apply(
        lambda row: check_if_pfams_are_in_subset(row, subset_pfams),
        axis=1
    )
    subset = data[mask]
    print(f"{data.shape}->{subset.shape}")
    subset.to_parquet("dataset/subsets/complicated_subset/data.parquet")
    df_to_fasta(subset,"dataset/subsets/complicated_subset/data.fasta")


if __name__ == '__main__':
    main()
