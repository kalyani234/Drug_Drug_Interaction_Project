import pandas as pd
from Bio import SeqIO

# File paths
fasta_path = "data/processed/mapped_protein_sequences.fasta"
drug_targets_path = "data/processed/corrected_mapped_drug_targets.csv"
output_mapped_targets_path = "data/processed/final_mapped_drug_targets.csv"

# Load FASTA and create mapping from ProtBERT IDs to keys in protein_embeddings.pt
def create_protbert_mapping(fasta_path):
    protbert_mapping = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        description_parts = record.description.split("|")
        if len(description_parts) > 2:
            protbert_id = description_parts[2]  # e.g., THRB_HUMAN
            full_id = record.id  # e.g., sp|P00734|THRB_HUMAN
            protbert_mapping[protbert_id] = full_id
    return protbert_mapping

# Map drug targets using the generated mapping
def map_drug_targets_with_embeddings(drug_targets_path, protbert_mapping, output_path):
    drug_targets = pd.read_csv(drug_targets_path)
    drug_targets["embedding_key"] = drug_targets["protbert_id"].map(protbert_mapping)
    unmapped = drug_targets[drug_targets["embedding_key"].isnull()]
    print(f"Number of unmapped proteins: {len(unmapped)}")
    print(f"Sample unmapped proteins:\n{unmapped.head()}")
    # Save mapped drug targets
    drug_targets.to_csv(output_path, index=False)
    print(f"Final mapped drug targets saved to {output_path}")

# Run mapping process
protbert_mapping = create_protbert_mapping(fasta_path)
map_drug_targets_with_embeddings(drug_targets_path, protbert_mapping, output_mapped_targets_path)
