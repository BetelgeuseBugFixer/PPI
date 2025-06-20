import copy
import json
import os
import pickle
import random
import sys
from collections import defaultdict, Counter

from dataset.analyse_pfams import get_data
from dataset.create_subset import get_clusters

SPLIT = [0.7, 0.10, 0.20]
MAX_ATTEMPTS = 10


def save_splits(split_file, train_ids, val_ids, test_ids):
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, 'w') as f:
        json.dump({
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }, f)


def get_protein_ids_from_fasta(fasta_path):
    protein_ids = set()
    with open(fasta_path) as fasta:
        for line in fasta:
            if line.startswith(">"):
                protein_ids.add(line[1:].rstrip())
    return protein_ids


def get_subset(data, protein_ids_in_subset):
    return data[data.index.isin(protein_ids_in_subset)]


def get_all_pfams_in_data(data_subset):
    all_pfams = set(pfam for row in data_subset["pfam_tensor"] for pfam in row)
    all_pfams.discard(None)
    return all_pfams


def get_pfam_to_cluster(protein_to_cluster, data_subset):
    pfam_to_cluster = defaultdict(set)
    for protein_id, row in data_subset.iterrows():
        cluster = protein_to_cluster[protein_id]
        pfams = set(row["pfam_tensor"])
        pfams.discard(None)
        for pfam in pfams:
            pfam_to_cluster[pfam].add(cluster)
    return pfam_to_cluster


def get_protein_ids_to_cluster(cluster_tsv_path):
    protein_to_cluster = {}
    with open(cluster_tsv_path, "r") as cluster_tsv:
        for line in cluster_tsv:
            if not line:
                continue
            fields = line.split("\t")
            protein_to_cluster[fields[1].strip()] = fields[0].strip()

    return protein_to_cluster


def get_split_deviation(current_split, split_percentages):
    all_samples = sum(current_split)
    deviation = 0
    for i, split_set in enumerate(current_split):
        deviation += abs((split_set / all_samples) - split_percentages[i])
    return deviation


def assign_cluster(cluster_size, number_of_protein_split, split_percentages):
    # since a split is scored in the difference in percentages, it cannot be worse than a deviation of 1 for each field
    best_deviation = len(split_percentages) + 1
    best_index = -1
    # assign cluster to each set and find best score
    for i in range(len(split_percentages)):
        current_split = number_of_protein_split.copy()
        current_split[i] += cluster_size
        deviation = get_split_deviation(current_split, split_percentages)
        if deviation < best_deviation:
            best_deviation = deviation
            best_index = i
    return best_index


def count_pfams(pfam_tensor_col):
    pfam_counter = Counter()
    for pfam_tensor in pfam_tensor_col:
        pfam_counter.update(pfam_tensor)
    return pfam_counter


def filter_subset_by_proteins(data_subset, final_protein_list):
    return data_subset[data_subset.index.isin(final_protein_list)]


def count_pfam_occurrences(new_subset):
    pfam_counter = Counter()
    for pfam_tensor in new_subset["pfam_tensor"]:
        pfam_counter.update(pfam_tensor)
    return pfam_counter


def main():
    cluster_dict = get_clusters("dataset/subsets/complicated_subset/cluster_cluster.tsv")
    protein_to_cluster = get_protein_ids_to_cluster("dataset/subsets/complicated_subset/cluster_cluster.tsv")
    data = get_data()
    protein_ids_in_subset = get_protein_ids_from_fasta("dataset/subsets/complicated_subset/data.fasta")
    data_subset = get_subset(data, protein_ids_in_subset)
    pfam_to_cluster_dict = get_pfam_to_cluster(protein_to_cluster, data_subset)
    # filter pfams with to little clusters
    while True:
        pfams_to_filter = set([k for k, v in pfam_to_cluster_dict.items() if len(v) < 3])
        if not pfams_to_filter:
            break
        # delete all pfam entries from the list and also save all clusters that need to be removed
        clusters_to_remove = set()
        for pfam_id in pfams_to_filter:
            clusters_to_remove |= pfam_to_cluster_dict[pfam_id]
            del pfam_to_cluster_dict[pfam_id]

        for pfam, pfam_clusters in pfam_to_cluster_dict.items():
            pfam_clusters -= clusters_to_remove

    # start splitting
    # prioritized_pfams = ["PF22752.1"," PF22451.2"]
    prioritized_pfams = []
    sorted_pfams = []
    current_split = {}
    # make extra dict so it can be edited and make it still
    current_pfam_to_cluster_dict = {}
    for attempt in range(MAX_ATTEMPTS):
        current_pfam_to_cluster_dict = copy.deepcopy(pfam_to_cluster_dict)
        sorted_pfams = sorted(current_pfam_to_cluster_dict.keys(),
                              key=lambda k: len([current_pfam_to_cluster_dict[k]]))

        # remove hard pfams and add it at the first index
        random.shuffle(prioritized_pfams)
        for pfam in prioritized_pfams:
            sorted_pfams.remove(pfam)
            sorted_pfams.insert(0, pfam)

        retry = False
        current_split = {}
        for pfam in sorted_pfams:
            if retry:
                break
            pfam_clusters = current_pfam_to_cluster_dict[pfam]
            # determine already made splits with clusters that are also on this pfam
            # saves the number of proteins in each set
            number_of_protein_split = [0, 0, 0]
            pfam_clusters_to_remove = [pfam_cluster for pfam_cluster in pfam_clusters if pfam_cluster in current_split]
            for pfam_cluster in pfam_clusters_to_remove:
                if pfam_cluster in current_split:
                    number_of_protein_split[current_split[pfam_cluster]] += current_split[pfam_cluster]
                    pfam_clusters.remove(pfam_cluster)
            # check if the remaining split is even possible
            missing_splits = [split_index for split_index in range(len(SPLIT)) if
                              number_of_protein_split[split_index] == 0]
            split_not_possible = len(missing_splits) > len(pfam_clusters)
            if split_not_possible:
                print(f"can not resolve pfam {pfam}")
                if attempt == MAX_ATTEMPTS - 1:
                    print("reached max attempts, will not be resolved")
                else:
                    if pfam not in prioritized_pfams:
                        prioritized_pfams.append(pfam)
                    print(f"starting new attempt: {attempt + 2}/{MAX_ATTEMPTS}")
                    retry = True
                    break

            # split remaining clusters according to the cluster size
            sorted_clusters = sorted(pfam_clusters, key=lambda pfam_cluster: len(cluster_dict[pfam_cluster]),
                                     reverse=True)
            # save the assignment to a list  first to check if any set is empty
            cluster_assignments = []
            for cluster in sorted_clusters:
                # greedily calculate best split
                cluster_size = len(cluster_dict[cluster])
                current_cluster_assignment = assign_cluster(cluster_size, number_of_protein_split, SPLIT)
                cluster_assignments.append(current_cluster_assignment)
                number_of_protein_split[current_cluster_assignment] += cluster_size

            # check if every set has an assigned cluster
            if not split_not_possible:
                missing_splits = [split_index for split_index in range(len(SPLIT)) if
                                  number_of_protein_split[split_index] == 0]
                if missing_splits:
                    # check if we have enough free clusters to fix
                    missing_splits.sort(key=lambda missing_split_index: SPLIT[missing_split_index])
                    for i, current_cluster_assignment in enumerate(missing_splits):
                        cluster_assignments[-i] = current_cluster_assignment

            # finalize split
            for i in range(len(cluster_assignments)):
                current_split[sorted_clusters[i]] = cluster_assignments[i]

        if not retry:
            print("successfully created split")
            break

    split_by_protein = [[], [], []]
    for cluster, current_cluster_assignment in current_split.items():
        split_by_protein[current_cluster_assignment] += cluster_dict[cluster]

    num_of_proteins = len(split_by_protein[0]) + len(split_by_protein[1]) + len(split_by_protein[2])
    print(num_of_proteins)
    print(len(split_by_protein[0]) / num_of_proteins)
    print(len(split_by_protein[1]) / num_of_proteins)
    print(len(split_by_protein[2]) / num_of_proteins)

    final_protein_list = split_by_protein[0] + split_by_protein[1] + split_by_protein[2]
    new_subset = filter_subset_by_proteins(data_subset, final_protein_list)
    pfam_counts = count_pfam_occurrences(new_subset)

    with open("Dataset/subsets/complicated_subset/pfam_counts.pkl", "wb") as f:
        pickle.dump(pfam_counts, f)

    save_splits("dataset/subsets/complicated_subset/split.json", split_by_protein[0], split_by_protein[1],
                split_by_protein[2])


if __name__ == '__main__':
    main()
