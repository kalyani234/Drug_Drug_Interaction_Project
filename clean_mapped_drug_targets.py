import pandas as pd

# File paths
drug_targets_path = "data/processed/mapped_drug_targets.csv"
corrected_drug_targets_path = "data/processed/corrected_mapped_drug_targets.csv"

# Load the drug targets file
print("Loading mapped drug targets...")
drug_targets = pd.read_csv(drug_targets_path)

# Clean the `protbert_id` column
print("Cleaning ProtBERT IDs...")
drug_targets["protbert_id"] = drug_targets["protbert_id"].str.split().str[0]

# Validate the cleaned data
print("Sample cleaned ProtBERT IDs:", drug_targets["protbert_id"].head())

# Save the corrected file
print(f"Saving corrected drug targets to {corrected_drug_targets_path}...")
drug_targets.to_csv(corrected_drug_targets_path, index=False)
print("Corrected drug targets saved successfully.")
