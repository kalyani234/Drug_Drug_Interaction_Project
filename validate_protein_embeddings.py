import pandas as pd
import torch

# Paths
mapped_targets_path = "data/processed/final_mapped_drug_targets.csv"
simplified_embeddings_path = "data/processed/simplified_protein_embeddings.pt"

# Load data
mapped_targets = pd.read_csv(mapped_targets_path)
protein_embeddings = torch.load(simplified_embeddings_path)

# Extract keys
protein_keys = list(protein_embeddings.keys())
print(f"Sample protein keys: {protein_keys[:10]}")

# Compare with `protbert_id` values
mapped_proteins = mapped_targets["protbert_id"].dropna().unique()
missing_proteins = [pid for pid in mapped_proteins if pid not in protein_keys]

print(f"Number of mapped proteins: {len(mapped_proteins)}")
print(f"Number of missing proteins: {len(missing_proteins)}")
print(f"Missing proteins (sample): {missing_proteins[:10]}")
