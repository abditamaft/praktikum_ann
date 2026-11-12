"""Eksperimen mahasiswa: 5 konfigurasi ANN (Bagian 7 modul).
Jalankan:  python eksperimen.py   (tambahkan --no-show untuk tanpa jendela grafik)"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from common import FEATURES, OUTPUT_DIR, build_model, load_dataset

SHOW = "--no-show" not in sys.argv
NAMES = ["setosa", "versicolor", "virginica"]
KONFIG = [
    (1, (5,), "relu"),
    (2, (10,), "relu"),
    (3, (20,), "relu"),
    (4, (10, 5), "relu"),
    (5, (10,), "tanh"),
]

df = load_dataset()
X, y = df[FEATURES], df["species_id"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

rows, curves, cms = [], {}, {}
for no, hidden, act in KONFIG:
    m = build_model(hidden, act).fit(X_train, y_train)
    yp = m.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, yp, average="macro", zero_division=0)
    nama = f"{hidden} {act}"
    rows.append({"eksperimen": no, "hidden_layer_sizes": str(hidden),
                 "aktivasi": act,
                 "accuracy": round(accuracy_score(y_test, yp), 4),
                 "precision_macro": round(p, 4), "recall_macro": round(r, 4),
                 "f1_macro": round(f, 4),
                 "iterasi": m.named_steps["ann"].n_iter_})
    curves[nama] = m.named_steps["ann"].loss_curve_
    cms[nama] = confusion_matrix(y_test, yp, labels=[0, 1, 2])

hasil = pd.DataFrame(rows)
print(hasil.to_string(index=False))
hasil.to_csv(OUTPUT_DIR / "hasil_eksperimen.csv", index=False)

# Loss curve semua konfigurasi
plt.figure(figsize=(7, 4.5))
for nama, c in curves.items():
    plt.plot(c, label=nama)
plt.title("Perbandingan Loss Curve")
plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eksperimen_loss_curve.png", dpi=150)

# Confusion matrix semua konfigurasi
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (nama, cm) in zip(axes.ravel(), cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(nama)
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eksperimen_confusion_matrix.png", dpi=150)
if SHOW:
    plt.show()
print(f"\nTersimpan di: {OUTPUT_DIR}")
