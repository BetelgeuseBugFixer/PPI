from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

cluster_file="dataset/cluster/mmseqs_res_cluster.tsv"


def parse_clusters(tsv_file):
    cluster_counter = defaultdict(int)
    with open(tsv_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                cluster_counter[fields[0]]+=1

    return list(cluster_counter.values())


def plot_cluster_sizes(sizes, output_file):
    """Generate histogram of cluster sizes"""
    plt.figure(figsize=(10, 6))

    # Calculate histogram data with dynamic binning
    max_size = max(sizes)
    bins = np.logspace(0, np.log10(max_size + 1), 50) if max_size > 100 else max_size

    plt.hist(sizes, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)

    # Configure plot
    plt.title(f'Cluster Size Distribution (n={len(sizes)} clusters)', fontsize=14)
    plt.xlabel('Cluster Size (number of sequences)', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Use log scales for large size ranges
    if max_size > 100:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Cluster Size (log scale)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to: {output_file}")


def main():
    sizes = parse_clusters(cluster_file)

    print(f"Found {len(sizes)} clusters")
    print(f"Largest cluster: {max(sizes)} sequences")
    print(f"Smallest cluster: {min(sizes)} sequences")
    print(f"Median cluster size: {np.median(sizes):.1f} sequences")
    print(Counter(sizes))
    plot_cluster_sizes(sizes, "dataset/cluster/sizes.jpg")


if __name__ == '__main__':
    main()