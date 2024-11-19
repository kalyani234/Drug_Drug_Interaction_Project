import pandas as pd
import torch
import h5py

# Paths
ddi_labels_path = "data/input/balanced_ddi_labels.parquet"
drug_embeddings_path = "data/processed/drug_embeddings.pt"
output_features_path = "data/processed/drug_pair_features.h5"

# Load drug embeddings
print("Loading drug embeddings...")
drug_embeddings = torch.load(drug_embeddings_path)
print(f"Loaded {len(drug_embeddings)} drug embeddings.")

# Function to process a batch of drug pairs
def process_batch(batch, drug_embeddings):
    features = []
    labels = []
    skipped_pairs = 0
    for _, row in batch.iterrows():
        drug1 = row["drug1"]
        drug2 = row["drug2"]
        label = row["label"]
        if drug1 in drug_embeddings and drug2 in drug_embeddings:
            # Concatenate and flatten the embeddings of the two drugs
            feature_vector = torch.cat((drug_embeddings[drug1], drug_embeddings[drug2])).flatten()
            features.append(feature_vector.numpy())  # Convert to NumPy for HDF5
            labels.append(label)
        else:
            skipped_pairs += 1
    return features, labels, skipped_pairs

# Read entire dataset and split into chunks manually
print("Reading DDI labels...")
ddi_labels = pd.read_parquet(ddi_labels_path, engine="fastparquet")
chunk_size = 50000
num_chunks = len(ddi_labels) // chunk_size + 1

print(f"Processing {len(ddi_labels)} records in {num_chunks} chunks...")

# Prepare HDF5 file for incremental saving
with h5py.File(output_features_path, "w") as f:
    # Assume each drug embedding is 1024-dimensional, so the pair is 2048
    f.create_dataset("features", shape=(0, 2048), maxshape=(None, 2048), dtype="float32", compression="gzip")
    f.create_dataset("labels", shape=(0,), maxshape=(None,), dtype="int", compression="gzip")

    total_processed = 0
    total_skipped = 0

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(ddi_labels))
        chunk = ddi_labels.iloc[start_idx:end_idx]

        batch_features, batch_labels, skipped_pairs = process_batch(chunk, drug_embeddings)

        # Incrementally save to HDF5
        if batch_features:  # Only save if there are valid features
            f["features"].resize((f["features"].shape[0] + len(batch_features)), axis=0)
            f["features"][-len(batch_features):] = batch_features
            f["labels"].resize((f["labels"].shape[0] + len(batch_labels)), axis=0)
            f["labels"][-len(batch_labels):] = batch_labels

        total_skipped += skipped_pairs
        total_processed += len(batch_features)

        print(f"Processed {total_processed} records, skipped {total_skipped} pairs so far.")

print("Features and labels saved successfully.")
print(f"Total processed pairs: {total_processed}")
print(f"Total skipped pairs: {total_skipped}")
