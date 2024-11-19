import pandas as pd
from Bio import SeqIO

# File paths
drug_targets_path = "data/input/drug_targets.csv"
corrected_fasta_path = "data/processed/mapped_protein_sequences.fasta"
output_mapped_targets = "data/processed/mapped_drug_targets.csv"

# Step 1: Extract valid protein mappings from FASTA
def load_protein_mappings(fasta_path):
    mappings = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description.split("|")
        if len(header) > 2:
            uniprot_id = header[1]  # UniProt ID
            protbert_id = header[2]  # ProtBERT ID
            mappings[uniprot_id] = protbert_id
    return mappings

# Step 2: Map drug targets to valid protein IDs
def map_drug_targets(drug_targets, mappings):
    drug_targets["protbert_id"] = drug_targets["protein_id"].map(mappings)
    unmapped = drug_targets[drug_targets["protbert_id"].isnull()]
    print(f"Number of unmapped proteins: {len(unmapped)}")
    print("Sample unmapped proteins:", unmapped.head())
    return drug_targets.dropna(subset=["protbert_id"])

# Main function
def main():
    print("Loading data...")
    drug_targets = pd.read_csv(drug_targets_path)
    mappings = load_protein_mappings(corrected_fasta_path)
    print(f"Loaded {len(mappings)} protein mappings.")

    print("Mapping protein IDs...")
    mapped_targets = map_drug_targets(drug_targets, mappings)

    print(f"Saving mapped drug targets to {output_mapped_targets}...")
    mapped_targets.to_csv(output_mapped_targets, index=False)
    print("Mapped drug targets saved successfully.")

if __name__ == "__main__":
    main()
