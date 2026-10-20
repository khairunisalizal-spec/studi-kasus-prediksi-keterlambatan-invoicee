# Studi Kasus: Analisis & Prediksi Keterlambatan Pembayaran Invoice

Studi kasus berdasarkan materi kuliah tamu *Make Sense of Data with Analysis and AI* tentang penerapan analisis data dan AI di dunia industri.

**Kasus:** PT Maju Bersama (fiktif), pendapatannya naik, tetapi laba turun dan piutang meningkat. Manajemen ingin tahu apa yang terjadi, mengapa, siapa yang berpotensi telat bayar, dan tindakan apa yang harus diambil.

Data yang dipakai adalah data sintetis buatan sendiri, dibuat sengaja "berantakan" (nilai kosong, duplikat, kategori tidak konsisten, transaksi mustahil, diskon tidak wajar) untuk menggambarkan kondisi data di lapangan.

## Hipotesis

| Kode | Hipotesis |
|---|---|
| H1 | Margin turun karena diskon makin besar |
| H2 | Penagihan melambat sehingga piutang naik |
| H3 | Pelanggan dengan tempo panjang, diskon besar, atau skor risiko tinggi lebih sering telat bayar |

## Kerangka CRISP-DM

| Tahap | Implementasi |
|---|---|
| 1. Business Understanding | Menekan piutang dengan memprediksi invoice yang berpotensi terlambat dibayar |
| 2. Data Understanding | Cek ukuran, nilai kosong, duplikat, variasi kategori, cek granularitas sebelum *join* |
| 3. Data Preparation | `prepare()`: hapus duplikat, standarisasi region, isi nilai kosong dari master pelanggan, buang transaksi mustahil dan diskon di atas 30%, buat fitur turunan |
| 4. Modeling | Baseline, Logistic Regression, dan Random Forest dengan penanganan data tidak seimbang |
| 5. Evaluation | Akurasi, recall, presisi, AUC (split berdasarkan waktu), demo *data leakage* |
| 6. Deployment (simulasi) | Daftar prioritas penagihan dan simulasi batas diskon |

## Empat Level Analitik

| Level | Pertanyaan | Output |
|---|---|---|
| Deskriptif | Apa yang terjadi? | `output/1_deskriptif.png`, `output/1_deskriptif_per_kuartal.csv` |
| Diagnostik | Mengapa terjadi? | `output/2_diagnostik.png` (telat bayar per segmen, termin, diskon, wilayah) |
| Prediktif | Apa yang akan terjadi? | `output/3_evaluasi_model.csv`, `output/3_feature_importance.png` |
| Preskriptif | Apa yang harus dilakukan? | `output/4_rekomendasi_tindakan.csv` |

## Cara Menjalankan

```bash
pip install -r requirements.txt
python analisis_maju_bersama.py
```

Semua hasil tersimpan di folder `output/`.

## Catatan Penting

- **Data tidak seimbang:** hanya sekitar 10% invoice yang terlambat. Baseline "selalu tepat" mendapat akurasi tinggi (sekitar 0,88) tetapi recall 0, sehingga akurasi saja menyesatkan.
- **Data leakage:** model yang memakai `days_late` tampak nyaris sempurna (AUC sekitar 1,0) padahal fitur itu baru diketahui setelah pembayaran terjadi. Model final tidak memakainya. Data juga dipisah berdasarkan waktu, bukan acak.
- **Performa model moderat** (AUC sekitar 0,64 sampai 0,67). Ini wajar karena keterlambatan bergantung pada faktor yang tidak ada di data (kondisi kas pelanggan, dsb.). Model dipakai sebagai alat bantu prioritas, bukan pengambil keputusan.
- **Peran manusia:** hasil model dan rekomendasi tetap harus diverifikasi secara kritis. Ambang risiko pada tahap preskriptif (0,3 dan 0,5) adalah asumsi yang perlu disesuaikan dengan kebijakan perusahaan.
- **Simulasi diskon** mengasumsikan volume penjualan tidak berubah, sehingga risiko kehilangan pelanggan tidak ikut dihitung.

## Pengembangan Lanjutan

Bangun alur kerja yang sama di KNIME dan bandingkan hasilnya, tambah fitur riwayat keterlambatan pelanggan (hanya dari invoice sebelumnya agar tidak bocor), coba Gradient Boosting, atau lakukan *hyperparameter tuning*.

## Referensi

Materi kuliah tamu *Make Sense of Data with Analysis and AI*, Informatics Practitioner Lecture Series 2026.
