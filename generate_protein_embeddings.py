import torch
from transformers import BertTokenizer, BertModel
from Bio import SeqIO

# Paths
fasta_file = "data/processed/mapped_protein_sequences.fasta"
output_embeddings_path = "data/processed/protein_embeddings.pt"

# Load ProtBERT tokenizer and model
print("Loading ProtBERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
model = BertModel.from_pretrained("Rostlab/prot_bert")
model.eval()  # Set model to evaluation mode
print("ProtBERT loaded successfully.")

# Function to generate embedding for a single protein sequence
def generate_protein_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():  # Disable gradient computation
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

# Load protein sequences from FASTA and generate embeddings
def generate_protein_embeddings(fasta_file, output_path):
    protein_embeddings = {}
    total_sequences = 0

    print(f"Reading sequences from {fasta_file}...")
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        sequence = str(record.seq)

        try:
            # Generate embedding
            embedding = generate_protein_embedding(sequence)
            protein_embeddings[protein_id] = embedding
            total_sequences += 1

            # Logging progress
            if total_sequences % 100 == 0:
                print(f"Processed {total_sequences} proteins...")

        except Exception as e:
            print(f"Error processing {protein_id}: {e}")

    # Save embeddings
    print(f"Saving {len(protein_embeddings)} protein embeddings to {output_path}...")
    torch.save(protein_embeddings, output_path)
    print("Protein embeddings saved successfully.")

# Main function
if __name__ == "__main__":
    generate_protein_embeddings(fasta_file, output_embeddings_path)
