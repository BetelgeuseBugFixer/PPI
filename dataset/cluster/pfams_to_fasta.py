import sys
from collections import defaultdict

pfam_seed_alignment_file_path = "dataset/Pfam-A.seed"
pfam_fasta_output_path = "dataset/pfams.fasta"


def get_cons_seq(family_alignment_seqs: list) -> str:
    cons_seq = ""
    for i in range(len(family_alignment_seqs[0])):
        aa_counter = defaultdict(int)
        for seq in family_alignment_seqs:
            aa_counter[seq[i]] += 1
        cons_aa = max(aa_counter, key=aa_counter.get)
        if cons_aa != ".":
            cons_seq += cons_aa
    return cons_seq


def main():
    with open(pfam_fasta_output_path, "w") as pfam_fasta_output:
        with open(pfam_seed_alignment_file_path, "r") as pfam_seed_alignment_file:
            current_family_id = ""
            family_alignment_seqs = defaultdict(str)
            for line in pfam_seed_alignment_file.readlines():
                if line.startswith("#=GF AC"):
                    current_family_id = line.split(maxsplit=3)[2]
                elif line.startswith("#"):
                    continue
                elif line.startswith("//"):
                    cons_seq = get_cons_seq(list(family_alignment_seqs.values()))
                    pfam_fasta_output.write(f">{current_family_id}\n{cons_seq}\n")
                    family_alignment_seqs = defaultdict(str)
                    current_family_id = ""
                else:
                    seq_id_and_alignment = line.split(maxsplit=2)
                    family_alignment_seqs[seq_id_and_alignment[0]] += seq_id_and_alignment[1]


if __name__ == '__main__':
    main()
