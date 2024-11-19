import pandas as pd
import requests
import time

# Paths
drug_targets_path = "data/input/drug_targets.csv"  # Input CSV file
output_fasta_path = "data/processed/filtered_protein_sequences.fasta"  # Output FASTA file
missing_ids_path = "data/logs/missing_protein_ids.txt"  # Missing protein IDs file (optional for retrying)

# UniProt URL template for fetching FASTA
uniprot_url = "https://www.uniprot.org/uniprot/{}.fasta"


# Step 1: Extract Unique Protein IDs
def extract_protein_ids(drug_targets_path):
    print("Extracting protein IDs from drug_targets.csv...")
    drug_targets = pd.read_csv(drug_targets_path)
    unique_protein_ids = drug_targets["protein_id"].unique()
    print(f"Total unique protein IDs: {len(unique_protein_ids)}")
    return unique_protein_ids


# Step 2: Fetch Protein Sequences from UniProt
def fetch_protein_sequences(protein_ids):
    print("Fetching protein sequences from UniProt...")
    fetched_sequences = []
    missing_ids = []

    for pid in protein_ids:
        url = uniprot_url.format(pid)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                fetched_sequences.append(response.text)
                print(f"Fetched: {pid}")
            else:
                print(f"Error fetching: {pid}")
                missing_ids.append(pid)
        except Exception as e:
            print(f"Exception fetching {pid}: {e}")
            missing_ids.append(pid)

        # Optional: Add a small delay to avoid rate limiting
        time.sleep(0.5)

    return fetched_sequences, missing_ids


# Step 3: Save Protein Sequences to FASTA
def save_to_fasta(sequences, output_fasta_path):
    print(f"Saving protein sequences to {output_fasta_path}...")
    with open(output_fasta_path, "w") as fasta_file:
        fasta_file.write("\n".join(sequences))
    print("Protein sequences saved successfully.")


# Step 4: Save Missing Protein IDs (if any)
def save_missing_ids(missing_ids, missing_ids_path):
    if missing_ids:
        print(f"Saving missing protein IDs to {missing_ids_path}...")
        with open(missing_ids_path, "w") as file:
            file.write("\n".join(missing_ids))
        print("Missing protein IDs saved successfully.")
    else:
        print("No missing protein IDs.")


# Main Function
def main():
    # Extract protein IDs
    protein_ids = extract_protein_ids(drug_targets_path)

    # Fetch protein sequences
    protein_sequences, missing_ids = fetch_protein_sequences(protein_ids)

    # Save successfully fetched sequences to FASTA file
    save_to_fasta(protein_sequences, output_fasta_path)

    # Save missing protein IDs for retrying if necessary
    save_missing_ids(missing_ids, missing_ids_path)


if __name__ == "__main__":
    main()
