import torch

# Load protein embeddings
protein_embeddings_path = "data/processed/protein_embeddings.pt"
output_path = "data/processed/simplified_protein_embeddings.pt"

protein_embeddings = torch.load(protein_embeddings_path)
simplified_embeddings = {}

# Simplify keys
for full_id, embedding in protein_embeddings.items():
    protbert_id = full_id.split("|")[-1]  # Extract ProtBERT ID
    simplified_embeddings[protbert_id] = embedding

# Save simplified embeddings
torch.save(simplified_embeddings, output_path)
print(f"Simplified protein embeddings saved to {output_path}")
print(f"Total simplified embeddings: {len(simplified_embeddings)}")
