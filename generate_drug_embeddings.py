import torch
import pandas as pd

# Paths
drug_targets_path = "data/processed/final_mapped_drug_targets.csv"
protein_embeddings_path = "data/processed/simplified_protein_embeddings.pt"
output_drug_embeddings_path = "data/processed/drug_embeddings.pt"

# Load data
print("Loading data...")
drug_targets = pd.read_csv(drug_targets_path)
protein_embeddings = torch.load(protein_embeddings_path)
print(f"Loaded {len(protein_embeddings)} protein embeddings.")

# Function to create drug embeddings by aggregating protein embeddings
def create_drug_embeddings(drug_targets, protein_embeddings):
    drug_embeddings = {}
    skipped_drugs = []
    for drug_id, group in drug_targets.groupby("drug_id"):
        protein_ids = group["protbert_id"].tolist()
        embeddings = [protein_embeddings[pid] for pid in protein_ids if pid in protein_embeddings]
        if embeddings:
            drug_embeddings[drug_id] = torch.mean(torch.stack(embeddings), dim=0)
        else:
            skipped_drugs.append(drug_id)
    return drug_embeddings, skipped_drugs

# Generate drug embeddings
print("Generating drug embeddings...")
drug_embeddings, skipped_drugs = create_drug_embeddings(drug_targets, protein_embeddings)
print(f"Generated {len(drug_embeddings)} drug embeddings.")
print(f"Skipped {len(skipped_drugs)} drugs due to missing protein embeddings.")
if skipped_drugs:
    print(f"Skipped drugs (sample): {skipped_drugs[:10]}")

# Save drug embeddings
print(f"Saving drug embeddings to {output_drug_embeddings_path}...")
torch.save(drug_embeddings, output_drug_embeddings_path)
print("Drug embeddings saved successfully.")
