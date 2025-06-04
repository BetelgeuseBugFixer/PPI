import json
import math
import pickle
from collections import Counter

import numpy as np
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


def plot_pfam_distribution(counter: Counter, title="Pfam Frequency Distribution", x_label="Pfam Occurrence Counts",
                           y_label="Number of Pfams", bin_size=50,
                           output_file="plot.jpeg", log_y=True):
    freq_of_freq = Counter(counter.values())

    binned = Counter()
    for count, num_pfam in freq_of_freq.items():
        bin_label = (count // bin_size) * bin_size
        binned[bin_label] += num_pfam

    x = sorted(binned.keys())
    y = [binned[bin_start] for bin_start in x]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, width=bin_size, align='edge',
            color='skyblue', edgecolor='black')

    plt.xlabel(f"{x_label} (binned, size={bin_size})")
    plt.ylabel(y_label)
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


def plot_binned_protein_lengths(length_counter, output_file, bin_size=10, log_y=False,x_lim=(0,1500)):
    lengths = np.array([int(k) for k in length_counter.keys()])
    counts = np.array([length_counter[str(k)] if str(k) in length_counter else length_counter[k] for k in lengths])

    max_length = lengths.max()
    if x_lim:
        max_length = min(max_length, x_lim[1])

    bins = np.arange(0, max_length + bin_size, bin_size)
    hist, edges = np.histogram(lengths, bins=bins, weights=counts)

    plt.figure(figsize=(12, 6))
    plt.bar(edges[:-1], hist, width=bin_size, align='edge', edgecolor='black', color='skyblue')
    plt.xlabel('Protein Length')
    plt.ylabel('Count')
    plt.title('Protein Length  Count')

    if log_y:
        plt.yscale('log')
        plt.ylabel('Count (log)')

    if x_lim:
        plt.xlim(x_lim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_counts(counter, title, filename, x_label, y_label, bin_size=None,log_y=False):
    # Convert keys to int (if they are strings)
    data_int = {int(k): v for k, v in counter.items()}

    if bin_size is None:
        # No binning: use each key directly
        x = sorted(data_int.keys())
        y = [data_int.get(i, 0) for i in x]
        x_labels = [str(i) for i in x]

    else:
        # Determine range of x
        all_x = sorted(data_int.keys())
        min_x, max_x = all_x[0], all_x[-1]
        # Compute number of bins
        num_bins = math.ceil((max_x - min_x + 1) / bin_size)
        binned_counts = []
        x_labels = []

        for b in range(num_bins):
            start = min_x + b * bin_size
            end = start + bin_size - 1
            # Sum counts whose keys fall into [start, end]
            total = sum(
                count for xi, count in data_int.items()
                if start <= xi <= end
            )
            binned_counts.append(total)
            # Label: "start–end"
            x_labels.append(f"{start}-{end}")

        # For plotting, use indices for bar positions
        x = list(range(len(binned_counts)))
        y = binned_counts

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Set x-ticks and labels
    ax.set_xticks(x)
    #ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Disable scientific notation on the y-axis
    ax.ticklabel_format(style='plain', axis='y')

    # If requested, switch to log scale on the y-axis
    if log_y:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

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


def main(load_data=True):
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
    delete_none_entries(number_of_pfams_per_protein)
    delete_none_entries(protein_lengths)
    get_number_of_pfams(pfams_per_aa)

    #plot_pfam_distribution(pfams_per_aa, title="pfams per aa counts", output_file="dataset/count_aa.jpeg", bin_size=100)
    # Bei x=1 und y=10000 bedeutet das, dass 10000 PFAMs genau einmal vorkommen.
    plot_pfam_distribution(pfams_per_protein, title="PFAM Occurrence Per Protein Frequency Distribution",
                           x_label="Number of Occurrences per PFAM", y_label="Number of PFAMs",
                           output_file="dataset/plots/protein.jpeg")
    plot_binned_protein_lengths(protein_lengths, "dataset/plots/protein_lengths.jpeg")
    plot_counts(number_of_pfams_per_protein, "Number of Pfams per Protein", "dataset/plots/number_pfam_per_protein.jpeg",
                "# of PFAMS in a protein", "Occurrence")


if __name__ == '__main__':
    main()
