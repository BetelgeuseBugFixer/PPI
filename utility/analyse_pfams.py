import _csv
import ast
import csv
import sys
from collections import Counter

from matplotlib import pyplot as plt


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


def count_pfams(path_to_dataset="dataset/dataset.csv") -> Counter:
    pfam_counter = Counter()
    with open(path_to_dataset, "r") as data_file:
        #reader = csv.DictReader(data_file)
        #try:
        #    for row in reader:
        #        pfam_counter += Counter(ast.literal_eval(row["pfam_tensor"]))
        #except _csv.Error:
        #    print(row)
        #    print(Counter(ast.literal_eval(get_field_from_csv_line(row.join(","),1))))
        for line in data_file.readlines()[1:]:
            pfam_counter += Counter(ast.literal_eval(get_field_from_csv_line(line,1)))
    return pfam_counter


def delete_none_entries(counter: Counter) -> None:
    number_of_none_entries = counter[None]
    number_of_aas = sum(counter.values())
    print(
        f"not annotated AAs: {number_of_none_entries}/{number_of_aas}->{(number_of_none_entries / number_of_aas) * 100}%")
    del counter[None]


def get_basic_statistics(counter: Counter) -> None:
    print(f"number of different pfams: {len(set(counter.keys()))}")
    plot_pfam_distribution(counter)


def plot_pfam_distribution(counter: Counter, title="Pfam Frequency Distribution", bin_size=50, output_file="plot.jpeg"):
    # Step 1: Count how often each count occurs
    freq_of_freq = Counter(counter.values())

    # Step 2: Bin the frequencies
    binned = Counter()
    for count, num_pfam in freq_of_freq.items():
        bin_label = (count // bin_size) * bin_size  # e.g. 27 -> 20 if bin_size=10
        binned[bin_label] += num_pfam

    # Step 3: Sort and prepare for plotting
    x = sorted(binned.keys())
    y = [binned[bin_start] for bin_start in x]
    bin_labels = [f"{i}-{i + bin_size - 1}" for i in x]

    # Step 4: Plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, y, width=0.8, color='skyblue', edgecolor='black')
    plt.xlabel(f"Pfam Occurrence Counts (binned, size={bin_size})")
    plt.ylabel("Number of Pfams")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    #counter = count_pfams("dataset/test.csv")
    counter = count_pfams()
    # print(counter)
    delete_none_entries(counter)
    get_basic_statistics(counter)


if __name__ == '__main__':
    main()

