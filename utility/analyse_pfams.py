import json
import math
import pickle
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

PFAMS_PER_AA_FILE = "dataset/pfams_per_aa.json"
PFAMS_PER_PROTEIN_FILE = "dataset/pfams_per_protein.json"
PROTEIN_LENGTHS_FILE = "dataset/protein_lengths.json"
NUMBER_OF_PFAMS_PER_PROTEIN_FILE = "dataset/number_of_pfams_per_protein.json"


def get_field_from_csv_line(line: str, field_index: int) -> str:
    current_index = 0
    in_right_field = current_index == field_index
    in_quotation_mark = False
    start = 0
    for line_index, char in enumerate(line):
        if char == '\"':
            in_quotation_mark = not in_quotation_mark
        elif not in_quotation_mark:
            if char == ',':
                if in_right_field:
                    return line[start: line_index].strip("\"")
                else:
                    current_index += 1
                    in_right_field = current_index == field_index
                    if in_right_field:
                        start = line_index + 1
    if in_right_field:
        return line[start:].strip("\"")
    raise RuntimeError


def calculate_percentage_of_multi_domain_proteins(data: pd.DataFrame) -> float:
    multi_domain_proteins = 0
    all_proteins = 0
    for pfams in data["pfam_tensor"].values:
        # remove None entries
        pfam_set = set(pfams)
        pfam_set.remove(None)
        if len(pfam_set) > 1:
            multi_domain_proteins += 1
        all_proteins += 1
    return multi_domain_proteins / all_proteins


def count_protein_length(data: pd.DataFrame) -> Counter:
    protein_lengths = []
    for pfams in data["pfam_tensor"].values:
        protein_lengths.append(len(pfams))
    return Counter(protein_lengths)


def count_number_of_pfams_per_protein(data: pd.DataFrame) -> Counter:
    num_of_pfams_per_proteins = []
    for pfams in data["pfam_tensor"].values:
        num_of_pfams_per_proteins.append(len(set(pfams)))
    return Counter(num_of_pfams_per_proteins)


def count_pfams_per_aa(data: pd.DataFrame) -> Counter:
    pfam_counter = Counter()
    for pfams in data["pfam_tensor"].values:
        pfam_counter += Counter(pfams)
    return pfam_counter


def count_pfams_per_protein(data: pd.DataFrame) -> Counter:
    pfam_counter = Counter()
    for pfams in data["pfam_tensor"].values:
        pfam_counter += Counter(set(pfams))
    return pfam_counter


def delete_none_entries(counter: Counter) -> None:
    del counter[None]


def print_percentage_of_nones(counter: Counter, title="None Entries") -> None:
    number_of_none_entries = counter[None]
    number_of_aas = sum(counter.values())
    print(
        f"{title}: {number_of_none_entries}/{number_of_aas}->{(number_of_none_entries / number_of_aas) * 100}%")


def get_number_of_pfams(counter: Counter) -> None:
    print(f"number of different pfams: {len(set(counter.keys()))}")


def plot_pfam_distribution(counter: Counter, title="Pfam Frequency Distribution", bin_size=50,
                           output_file="plot.jpeg", max_xticks=20, log_y=False):
    # Step 1: Count frequency of frequencies
    freq_of_freq = Counter(counter.values())

    # Step 2: Bin the counts
    binned = Counter()
    for count, num_pfam in freq_of_freq.items():
        bin_label = (count // bin_size) * bin_size
        binned[bin_label] += num_pfam

    # Handle empty data case
    if not binned:
        print("No data to plot")
        return

    # Step 3: Prepare data for plotting
    x = sorted(binned.keys())
    y = [binned[bin_start] for bin_start in x]

    # Calculate axis range and ticks
    min_bin = min(x)
    max_bin = max(x)
    total_range = max_bin + bin_size - min_bin
    """
    # Calculate optimal tick spacing
    if max_xticks < 2:
        max_xticks = 2  # Ensure at least 2 ticks for range

    desired_step = total_range / (max_xticks - 1)
    k = max(1, math.ceil(desired_step / bin_size))
    step = k * bin_size

    # Generate regular ticks
    tick_starts = []
    current = min_bin
    while current <= max_bin + bin_size:
        tick_starts.append(current)
        current += step

    # Ensure we don't exceed max_xticks
    while len(tick_starts) > max_xticks:
        k += 1
        step = k * bin_size
        tick_starts = []
        current = min_bin
        while current <= max_bin + bin_size:
            tick_starts.append(current)
            current += step

    tick_labels = [f"{s}" for s in tick_starts]
    """
    # Step 4: Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, width=bin_size, align='edge',
            color='skyblue', edgecolor='black')

    # Configure axis and labels
    # plt.xticks(tick_starts, tick_labels, rotation=45, ha='right')
    plt.xlabel(f"Pfam Occurrence Counts (binned, size={bin_size})")
    plt.ylabel("Number of Pfams")
    plt.title(title)

    # Handle logarithmic scale
    if log_y:
        plt.yscale('log')
        plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)
    else:
        plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def get_data(path_to_dataset="dataset/dataset.pkl") -> pd.DataFrame:
    with open(path_to_dataset, 'rb') as file:
        data = pickle.load(file)
    return data


def save_counter_to_json(counter: Counter, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(dict(counter), f)


def load_counter_from_json(filename: str) -> Counter:
    with open(filename, 'r') as f:
        data = Counter(json.load(f))
    data[None] = data["null"]
    del data["null"]
    return data


def main(load_data=False):
    if load_data:
        pfams_per_aa = load_counter_from_json(PFAMS_PER_AA_FILE)
        pfams_per_protein = load_counter_from_json(PFAMS_PER_PROTEIN_FILE)
        protein_lengths = load_counter_from_json(PROTEIN_LENGTHS_FILE)
        number_of_pfams_per_protein = load_counter_from_json(NUMBER_OF_PFAMS_PER_PROTEIN_FILE)
    else:
        # load data
        data = get_data()
        # get pfams per aa
        pfams_per_aa = count_pfams_per_aa(data)
        save_counter_to_json(pfams_per_aa, PFAMS_PER_AA_FILE)
        # get pfams per protein
        pfams_per_protein = count_pfams_per_protein(data)
        save_counter_to_json(pfams_per_protein, PFAMS_PER_PROTEIN_FILE)
        # get protein length distribution
        protein_lengths = count_protein_length(data)
        save_counter_to_json(protein_lengths, PROTEIN_LENGTHS_FILE)
        # count number of pfams per protein
        number_of_pfams_per_protein = count_number_of_pfams_per_protein(data)
        save_counter_to_json(number_of_pfams_per_protein, NUMBER_OF_PFAMS_PER_PROTEIN_FILE)

    delete_none_entries(pfams_per_protein)
    delete_none_entries(pfams_per_aa)
    get_number_of_pfams(pfams_per_aa)

    plot_pfam_distribution(pfams_per_aa, title="pfams per aa counts", output_file="count_aa.jpeg", log_y=True)
    plot_pfam_distribution(pfams_per_protein, title="pfams per protein counts", output_file="protein.jpeg")
    plot_pfam_distribution(protein_lengths, title="protein lengths", output_file="protein_lengths.jpeg")
    plot_pfam_distribution(number_of_pfams_per_protein, title="number of pfams per protein",
                           output_file="number_pfam_per_protein")


if __name__ == '__main__':
    main()
