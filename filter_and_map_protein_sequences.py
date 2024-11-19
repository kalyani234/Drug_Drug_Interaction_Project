import pandas as pd
from Bio import SeqIO

# Input paths
drug_targets_path = "data/input/drug_targets.csv"  # Input drug targets file
protein_fasta_path = "data/processed/filtered_protein_sequences.fasta"  # Original FASTA file

# Output paths
corrected_fasta_path = "data/processed/mapped_protein_sequences.fasta"  # Corrected FASTA file
mapping_csv_path = "data/processed/protein_id_mapping.csv"  # Mapping file (optional)


# Step 1: Extract unique protein IDs from drug_targets.csv
def extract_protein_ids(drug_targets_path):
    print("Extracting protein IDs from drug_targets.csv...")
    drug_targets = pd.read_csv(drug_targets_path)
    unique_protein_ids = drug_targets["protein_id"].unique()
    print(f"Total unique protein IDs: {len(unique_protein_ids)}")
    return unique_protein_ids


# Step 2: Create a mapping of UniProt IDs to ProtBERT IDs from FASTA file
def create_protein_id_mapping(protein_fasta_path):
    print("Creating UniProt to ProtBERT ID mapping...")
    protein_id_map = {}
    for record in SeqIO.parse(protein_fasta_path, "fasta"):
        header = record.description.split("|")  # Example: sp|P00734|THRB_HUMAN
        if len(header) > 2:  # Ensure the header contains UniProt and ProtBERT IDs
            uniprot_id = header[1]  # Extract P00734
            protbert_id = header[2]  # Extract THRB_HUMAN
            protein_id_map[uniprot_id] = protbert_id
    print(f"Created mapping for {len(protein_id_map)} protein IDs.")
    return protein_id_map


# Step 3: Filter and save corrected FASTA sequences
def save_corrected_fasta(protein_fasta_path, protein_ids, protein_id_map, corrected_fasta_path):
    print(f"Saving corrected FASTA to {corrected_fasta_path}...")
    corrected_sequences = []
    for record in SeqIO.parse(protein_fasta_path, "fasta"):
        header = record.description.split("|")
        if len(header) > 2:
            uniprot_id = header[1]  # P00734
            if uniprot_id in protein_ids:  # Check if UniProt ID is needed
                corrected_sequences.append(record)
    with open(corrected_fasta_path, "w") as corrected_file:
        SeqIO.write(corrected_sequences, corrected_file, "fasta")
    print(f"Saved {len(corrected_sequences)} corrected protein sequences.")


# Main function
def main():
    # Step 1: Extract protein IDs from drug_targets.csv
    protein_ids = extract_protein_ids(drug_targets_path)

    # Step 2: Create UniProt to ProtBERT ID mapping
    protein_id_map = create_protein_id_mapping(protein_fasta_path)

    # Step 3: Save corrected FASTA file
    save_corrected_fasta(protein_fasta_path, protein_ids, protein_id_map, corrected_fasta_path)

    # Optional: Save mapping to CSV for reference
    pd.DataFrame(list(protein_id_map.items()), columns=["UniProt_ID", "ProtBERT_ID"]).to_csv(
        mapping_csv_path, index=False
    )
    print(f"Protein ID mapping saved to {mapping_csv_path}.")


if __name__ == "__main__":
    main()
