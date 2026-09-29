"""
triage_classifier_train.py -- AUDITED VERSION

CHANGE FROM ORIGINAL (Phase 2): TfidfVectorizer was previously fit on
the full text series (train+val combined) BEFORE the train/val split,
so IDF weights for validation-set terms were influenced by their
occurrence in the validation set itself. Fixed: the vectorizer is now
fit on the train split's text only, and the val split is transformed
(not fit) using that already-fit vectorizer. The same fix is applied
inside cross_validate()'s fold loop, since it previously received the
already-globally-vectorized X and had the identical leak.

Nothing else changed: same MLP architecture, same 80/20 stratified
single-split protocol (disclosed as a secondary, less rigorously
validated evaluation per the paper's Limitations section), same fixed
40-epoch training with no val-based model selection, same output
filenames.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class TriageMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_text_and_labels(path="data/symptom_disease.csv"):
    df = pd.read_csv(path)
    assert "Urgency" in df.columns, "Add an Urgency column (0/1/2) before training."
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    text = df[symptom_cols].fillna("").agg(" ".join, axis=1)
    y = df["Urgency"].astype(int).values
    return text, y


def main(epochs=40, lr=1e-3, batch_size=32, max_features=500):
    text, y = load_text_and_labels()

    # Split FIRST, on raw text/labels -- vectorizer has not seen anything yet.
    text_train, text_val, y_train, y_val = train_test_split(
        text, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # FIX: fit on train text only, transform val with the already-fit vectorizer.
    vec = TfidfVectorizer(max_features=max_features)
    X_train = vec.fit_transform(text_train).toarray().astype(np.float32)
    X_val = vec.transform(text_val).toarray().astype(np.float32)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    n_classes = len(set(y))
    model = TriageMLP(X_train.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    n = X_train_t.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(1)
            val_out = model(X_val_t)
            val_loss = loss_fn(val_out, y_val_t).item()
            val_pred = val_out.argmax(1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            val_acc = (val_pred == y_val_t).float().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    import os
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(history).to_csv("results/triage_training_history.csv", index=False)

    model.eval()
    with torch.no_grad():
        final_pred = model(X_val_t).argmax(1).numpy()
    cm = confusion_matrix(y_val, final_pred, labels=sorted(set(y)))
    np.savetxt("results/triage_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print("Final val accuracy:", accuracy_score(y_val, final_pred))

    cross_validate(text, y, n_splits=5, seed=SEED, max_features=max_features)


def cross_validate(text, y, n_splits=5, seed=42, max_features=500):
    """FIX applied here too: vectorizer is fit fresh inside each fold on
    that fold's train portion only, not on a pre-vectorized global X."""
    text = pd.Series(text).reset_index(drop=True)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(text, y), 1):
        vec = TfidfVectorizer(max_features=max_features)
        X_tr = vec.fit_transform(text.iloc[tr_idx]).toarray().astype(np.float32)
        X_te = vec.transform(text.iloc[te_idx]).toarray().astype(np.float32)
        y_tr_fold, y_te_fold = y[tr_idx], y[te_idx]

        model = TriageMLP(X_tr.shape[1], n_classes=len(set(y)))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        X_tr_t = torch.tensor(X_tr)
        y_tr_t = torch.tensor(y_tr_fold, dtype=torch.long)
        X_te_t = torch.tensor(X_te)
        y_te_t = torch.tensor(y_te_fold, dtype=torch.long)

        for _ in range(40):
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t), y_tr_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
        accs.append(acc)
        print(f"Fold {fold}: acc={acc:.4f}")

    print(f"5-fold CV accuracy: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}")
    pd.DataFrame({"fold": range(1, n_splits + 1), "accuracy": accs}).to_csv(
        "results/triage_cv_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
