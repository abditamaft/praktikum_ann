# Praktikum ANN - Klasifikasi Spesies Bunga Iris (MLPClassifier)

Mata Kuliah: Sistem Cerdas - D4 Teknik Informatika

## Struktur Folder
```
ANN_Iris_Praktikum/
├── dataset/IRIS.csv                  # dataset (150 sampel, 4 fitur, 3 kelas)
├── common.py                         # load dataset + pipeline ANN
├── Program_ANN_Iris_Praktikum.py     # program utama (Langkah 1-8 + prediksi baru)
├── eksperimen.py                     # 5 eksperimen konfigurasi (Bagian 7)
├── requirements.txt
├── JAWABAN.md                        # jawaban pertanyaan evaluasi + catatan hasil
└── outputs/                          # hasil (dibuat otomatis saat program dijalankan)
```

## Cara Menjalankan
```bash
python -m pip install -r requirements.txt
python Program_ANN_Iris_Praktikum.py     # training, evaluasi, loss curve, prediksi
python eksperimen.py                      # perbandingan 5 konfigurasi
```
Tambahkan `--no-show` jika tidak ingin jendela grafik muncul (grafik tetap disimpan di `outputs/`).

## Catatan Dataset
`IRIS.csv` hanya memiliki kolom `species` (Iris-setosa, dst.) dan belum memiliki `species_id`.
Program otomatis membuat `species_id` (0=setosa, 1=versicolor, 2=virginica) dari kolom `species`,
sehingga file CSV asli tidak perlu diubah. Kolom `species` tidak digunakan sebagai fitur.

## Keluaran di folder outputs/
| File | Isi |
|---|---|
| classification_report.txt | precision, recall, F1 per kelas |
| confusion_matrix.png | confusion matrix data testing |
| loss_curve.png | kurva loss training |
| hasil_prediksi_iris.csv | prediksi 3 data bunga baru + probabilitas |
| hasil_eksperimen.csv | metrik 5 konfigurasi |
| eksperimen_loss_curve.png, eksperimen_confusion_matrix.png | perbandingan konfigurasi |
