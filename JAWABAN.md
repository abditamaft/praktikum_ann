# Jawaban Pertanyaan Evaluasi dan Catatan Hasil

> Hasil di bawah diperoleh dari menjalankan program dengan `random_state=42`.
> Angka dapat sedikit berbeda pada versi pustaka lain. Cocokkan dengan hasil di komputer Anda.

## Hasil Eksperimen (data testing 30 sampel)
| No | hidden_layer_sizes | Aktivasi | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|---|
| 1 | (5,) | relu | 0.9000 | 0.9141 | 0.9000 | 0.8950 |
| 2 | (10,) | relu | 0.8333 | 0.8571 | 0.8333 | 0.8289 |
| 3 | (20,) | relu | 0.9333 | 0.9444 | 0.9333 | 0.9327 |
| 4 | (10, 5) | relu | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| 5 | (10,) | tanh | 0.9000 | 0.9231 | 0.9000 | 0.8977 |

Pada konfigurasi utama modul ((10,), relu), setosa paling mudah diprediksi (precision 1.00, recall 0.90),
sedangkan versicolor paling sulit (recall 0.60) karena banyak tertukar dengan virginica. Hal ini wajar
karena ukuran versicolor dan virginica saling tumpang tindih, sementara setosa terpisah jelas.
Akurasi konfigurasi utama (0.83) relatif rendah karena `early_stopping=True` menyisihkan 10% data
training sebagai validasi dan training berhenti dini (sekitar 44 iterasi) pada dataset kecil. Hasil ini
sesuai pengaturan modul. Coba `early_stopping=False` untuk perbandingan dan catat di laporan.

## Prediksi 3 Data Baru
| sepal_length | sepal_width | petal_length | petal_width | Prediksi | Prob. tertinggi |
|---|---|---|---|---|---|
| 5.1 | 3.5 | 1.4 | 0.2 | setosa | 63.30% |
| 6.0 | 2.9 | 4.5 | 1.5 | versicolor | 37.58% |
| 6.5 | 3.0 | 5.8 | 2.2 | virginica | 48.62% |

Probabilitas relatif rendah karena model berhenti dini. Ketiga prediksi tetap sesuai spesies yang diharapkan.

## Mengapa data testing tidak boleh dipakai untuk training?
Data testing berfungsi menguji kemampuan model pada data yang belum pernah dilihat. Jika dipakai untuk
training, model sudah "mengetahui" jawabannya sehingga akurasi terlihat tinggi tetapi tidak mencerminkan
kemampuan generalisasi (data leakage).

## Pertanyaan Evaluasi
1. **Fitur input vs target:** fitur input adalah empat pengukuran (sepal_length, sepal_width, petal_length, petal_width) yang menjadi masukan model. Target adalah `species_id`, yaitu label yang ingin diprediksi.
2. **Mengapa `species` bukan fitur:** kolom itu adalah nama kelas (jawaban) itu sendiri. Memasukkannya sebagai fitur membuat model "mencontek" dan tidak belajar dari pengukuran bunga.
3. **`hidden_layer_sizes=(10,)`:** satu hidden layer dengan 10 neuron.
4. **StandardScaler:** menstandardisasi tiap fitur menjadi rata-rata 0 dan simpangan baku 1 agar skala fitur seragam dan training ANN stabil serta lebih cepat konvergen.
5. **`stratify=y`:** menjaga proporsi tiap kelas di data training dan testing sama dengan dataset asli (di sini 40:40:40 dan 10:10:10).
6. **`predict()` vs `predict_proba()`:** `predict()` menghasilkan satu kelas prediksi, sedangkan `predict_proba()` menghasilkan probabilitas untuk setiap kelas.
7. **Membaca confusion matrix 3 kelas:** baris = kelas aktual, kolom = kelas prediksi. Diagonal utama adalah prediksi benar; sel di luar diagonal adalah kesalahan (mis. baris versicolor kolom virginica = versicolor yang diprediksi virginica).
8. **Mengapa data baru melalui Pipeline yang sama:** agar StandardScaler yang dipelajari dari data training diterapkan juga ke data baru. Standardisasi terpisah akan membuat skala berbeda dari yang dipelajari model dan merusak prediksi.
