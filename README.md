
# Drug-Drug Interaction Prediction Project

This project focuses on predicting drug-drug interactions (DDIs) using machine learning (Random Forest) and deep learning (Neural Networks). It leverages pre-computed embeddings for drugs and proteins to create a robust feature set.

---

## **Project Structure**


```plaintext
├── data
│   ├── input
│   │   ├── balanced_ddi_labels.parquet
│   │   ├── ddi_labels.csv
│   │   └── drug_targets.csv
│   ├── processed
│       ├── drug_embeddings.pt
│       ├── drug_pair_features.h5
│       ├── protein_embeddings.pt
│       ├── mapped_protein_sequences.fasta
│       ├── protein_id_mapping.csv
│       └── simplified_protein_embeddings.pt
├── models
│   └── train_evaluate_models.py
├── results
│   ├── random_forest.pkl
│   ├── nn_model.pth
│   └── neural_network_pr_curve.png
├── scripts
│   ├── clean_mapped_drug_targets.py
│   ├── extract_protein_sequences.py
│   ├── filter_and_map_protein_sequences.py
│   ├── generate_drug_embeddings.py
│   ├── generate_drug_pair_features.py
│   ├── generate_features.py
│   ├── generate_protein_embeddings.py
│   ├── map_drug_targets_to_protein_ids.py
│   └── update_keys_protein_embeddings.py
```


## Steps to Reproduce

1. **Data Preprocessing**
   - Extract Protein Sequences: \`extract_protein_sequences.py\`
   - Filter and Map Sequences: \`filter_and_map_protein_sequences.py\`
   - Generate Embeddings:
     - Drugs: \`generate_drug_embeddings.py\`
     - Proteins: \`generate_protein_embeddings.py\`
   - Generate Pair Features: \`generate_drug_pair_features.py\`
   - Balance and Validate Data: \`generate_features.py\`

2. **Training and Evaluation**
   - Train and evaluate models using \`train_evaluate_models.py\`.

## Dependencies

Install required Python libraries:

```
pip install -r requirements.txt
```

## Outputs
- Random Forest Model: ```results/random_forest.pkl```
- Neural Network Model: ```results/nn_model.pth```
- PR Curve: ```results/neural_network_pr_curve.png```


