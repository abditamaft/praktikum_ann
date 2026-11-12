"""Praktikum ANN - Klasifikasi Spesies Bunga Iris (MLPClassifier).
Jalankan:  python Program_ANN_Iris_Praktikum.py
Tambahkan --no-show bila tidak ingin jendela grafik muncul."""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

from common import (FEATURES, NAMA_KELAS, OUTPUT_DIR, build_model,
                    load_dataset)

SHOW = "--no-show" not in sys.argv
NAMES = ["setosa", "versicolor", "virginica"]


def judul(teks):
    print("\n" + "=" * 60 + f"\n{teks}\n" + "=" * 60)


# Langkah 1 - Membaca dataset
judul("LANGKAH 1 - MEMBACA DATASET")
df = load_dataset()
print(df.head())
print()
df.info()
print("\nJumlah sampel per spesies:")
print(df["species"].value_counts())

# Langkah 2 - Memisahkan fitur dan target
judul("LANGKAH 2 - FITUR DAN TARGET")
X = df[FEATURES]
y = df["species_id"]
print("Fitur :", FEATURES)
print("Target: species_id (0=setosa, 1=versicolor, 2=virginica)")

# Langkah 3 - Membagi data training dan testing
judul("LANGKAH 3 - PEMBAGIAN DATA (80:20, stratify)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training:", len(X_train))
print("Testing :", len(X_test))

# Langkah 4 - Pipeline dan ANN
judul("LANGKAH 4 - PIPELINE DAN ANN")
model = build_model(hidden=(10,), activation="relu")
print(model)

# Langkah 5 - Training
judul("LANGKAH 5 - TRAINING")
model.fit(X_train, y_train)
print("Training selesai")

# Langkah 6 - Prediksi data testing
y_pred = model.predict(X_test)

# Langkah 7 - Evaluasi
judul("LANGKAH 7 - EVALUASI")
print("Akurasi:", accuracy_score(y_test, y_pred))
laporan = classification_report(y_test, y_pred, labels=[0, 1, 2],
                                target_names=NAMES)
print(laporan)
(OUTPUT_DIR / "classification_report.txt").write_text(laporan, encoding="utf-8")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NAMES).plot(
    cmap="Blues", values_format="d")
plt.title("Confusion Matrix ANN Iris")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
if SHOW:
    plt.show()
plt.close()

# Langkah 8 - Loss curve
ann = model.named_steps["ann"]
plt.plot(ann.loss_curve_)
plt.title("Loss Curve ANN Iris")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
if SHOW:
    plt.show()
plt.close()

# 6.1 Prediksi satu sampel
judul("6.1 PREDIKSI SATU SAMPEL")
data_baru = pd.DataFrame([{
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
}])
pred_id = int(model.predict(data_baru)[0])
prob = model.predict_proba(data_baru)[0]
print("Prediksi:", NAMA_KELAS[pred_id])
for i, nama in NAMA_KELAS.items():
    print(f"{nama}: {prob[i] * 100:.2f}%")

# 6.2 Prediksi beberapa sampel
judul("6.2 PREDIKSI BEBERAPA SAMPEL")
data_baru = pd.DataFrame([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.8, 2.2],
], columns=FEATURES)
pred = model.predict(data_baru)
prob = model.predict_proba(data_baru)
hasil = data_baru.copy()
hasil["prediksi_id"] = pred
hasil["prediksi_spesies"] = [NAMA_KELAS[int(v)] for v in pred]
hasil["probabilitas_tertinggi(%)"] = (prob.max(axis=1) * 100).round(2)
print(hasil.to_string())
hasil.to_csv(OUTPUT_DIR / "hasil_prediksi_iris.csv", index=False)

print(f"\nSemua keluaran tersimpan di folder: {OUTPUT_DIR}")
