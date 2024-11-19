import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import h5py
import joblib

# Paths
features_path = "data/processed/drug_pair_features.h5"
results_path = "results"
os.makedirs(results_path, exist_ok=True)

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


# PyTorch Dataset for Streaming Data
class DrugDataset(Dataset):
    def __init__(self, features_path, mode="train", train_split=0.1):
        self.file_path = features_path
        self.mode = mode
        self.train_split = train_split

        with h5py.File(self.file_path, "r") as f:
            self.total_samples = len(f["labels"])
            split_idx = int(self.total_samples * train_split)
            if self.mode == "train":
                self.indices = np.arange(0, split_idx)
            else:
                self.indices = np.arange(split_idx, self.total_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        with h5py.File(self.file_path, "r") as f:
            feature = f["features"][real_idx]
            label = f["labels"][real_idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Evaluate Model (Unified)
def evaluate_model(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"{model_name} Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")

    # Plot Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall_vals, precision_vals, label=f"{model_name} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(results_path, f"{model_name.lower()}_pr_curve.png"))
    plt.close()


# Train Random Forest (Optimized)
def train_random_forest(train_dataset, test_dataset, n_estimators=100):
    print("Training Random Forest (Optimized)...")

    # Load train and test data
    X_train, y_train = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    X_test, y_test = zip(*[test_dataset[i] for i in range(len(test_dataset))])

    X_train, X_test = torch.stack(X_train).numpy(), torch.stack(X_test).numpy()
    y_train, y_test = torch.tensor(y_train).numpy(), torch.tensor(y_test).numpy()

    # Check class distribution
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"Random Forest class distribution (train): {dict(zip(unique_classes, class_counts))}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,  # Limit depth for faster training
        max_features="sqrt",  # Use sqrt of features
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")

    # Evaluate Random Forest
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred, y_prob, "Random Forest")

    return rf


# Train Neural Network
def train_neural_network(train_loader, val_loader, input_size):
    print("Training Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    for epoch in range(10):  # Reduced epochs for faster training
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_prob = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                val_loss += criterion(outputs, labels).item()
                y_true.extend(labels.tolist())
                y_prob.extend(outputs.tolist())

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model


def evaluate_neural_network(model, test_loader):
    print("Evaluating Neural Network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            y_prob.extend(outputs.cpu().tolist())
            y_pred.extend((outputs > 0.5).cpu().int().tolist())
            y_true.extend(labels.cpu().tolist())

    evaluate_model(y_true, y_pred, y_prob, "Neural Network")


# Main Function
def main():
    train_dataset = DrugDataset(features_path, mode="train", train_split=0.1)
    test_dataset = DrugDataset(features_path, mode="test", train_split=0.1)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    print("Training and evaluating models...")

    # Train Random Forest (Optimized)
    rf_model = train_random_forest(train_dataset, test_dataset, n_estimators=100)
    joblib.dump(rf_model, os.path.join(results_path, "random_forest.pkl"))

    # Train and Evaluate Neural Network
    nn_model = train_neural_network(train_loader, test_loader, input_size=train_loader.dataset[0][0].shape[0])
    torch.save(nn_model.state_dict(), os.path.join(results_path, "nn_model.pth"))
    evaluate_neural_network(nn_model, test_loader)

    print("All models trained and evaluated successfully.")


if __name__ == "__main__":
    main()
