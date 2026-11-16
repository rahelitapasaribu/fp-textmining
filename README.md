# Klasifikasi Hate Speech Tweet Bahasa Indonesia

Aplikasi untuk mengklasifikasikan tweet berbahasa Indonesia ke dalam dua kategori: **Hate Speech** dan **Non-Hate Speech**. Aplikasi menggunakan preprocessing teks, TF-IDF, dan model Support Vector Machine (SVM), lalu menyediakan riwayat klasifikasi menggunakan SQLite.

## Fitur

- Mengklasifikasikan satu tweet melalui input teks.
- Mengklasifikasikan banyak tweet dari file CSV dan mengunduh hasilnya.
- Menampilkan label dan confidence score.
- Menyimpan, melihat, mengunduh, dan menghapus riwayat klasifikasi.
- Menampilkan metrik evaluasi model jika tersedia di `model/model_info.json`.

## Struktur proyek

```text
app/
  app.py                         # Aplikasi Streamlit
  riwayat_klasifikasi.db         # Database riwayat SQLite
dataset/
  raw/                           # Dataset awal dan kamus normalisasi
  processed/                     # Data hasil EDA, preprocessing, dan pembagian
model/
  best_svm_model.pkl             # Model SVM
  tfidf_vectorizer.joblib        # Vectorizer jika model bukan Pipeline
  model_info.json                # Metrik evaluasi model
notebook/
  data_loader_eda.ipynb          # Pemuatan data dan EDA
  preprocessing.ipynb            # Preprocessing
  splitting_data.ipynb           # Pembagian data
  svm.ipynb                      # Pelatihan dan evaluasi SVM
requirements.txt                 # Dependensi Python
```

## Persiapan dan menjalankan aplikasi

Pastikan Python sudah terpasang. Dari direktori utama proyek, jalankan perintah berikut di PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Streamlit akan menampilkan alamat lokal aplikasi di terminal; buka alamat tersebut di browser.

> Versi library di `requirements.txt` belum dikunci. Untuk memastikan hasil yang lebih konsisten antar komputer, gunakan versi dependensi yang sama dan uji aplikasi setelah pemasangan.

## Menggunakan aplikasi

1. Buka halaman **Dashboard**.
2. Pilih tab **Input Teks** untuk mengklasifikasikan satu tweet, atau **Upload File CSV** untuk memproses beberapa tweet.
3. Untuk CSV, pilih kolom yang berisi teks. Aplikasi akan menyarankan kolom bernama `tweet`, `text`, `teks`, atau `content` jika tersedia.
4. Lihat hasil klasifikasi dan confidence score. Hasil CSV dapat diunduh.
5. Buka **Riwayat Klasifikasi** untuk melihat atau mengelola proses sebelumnya.

Preprocessing aplikasi mencakup pembersihan teks, case folding, tokenisasi, normalisasi menggunakan kamus, penghapusan stopword, dan stemming. File kamus yang digunakan adalah `dataset/raw/new_kamusalay.csv`.

## Data dan model

Aplikasi membutuhkan file model `model/best_svm_model.pkl` dan kamus normalisasi `dataset/raw/new_kamusalay.csv`. Jika model yang digunakan bukan Pipeline TF-IDF + SVM, aplikasi juga membutuhkan `model/tfidf_vectorizer.joblib`.

Database SQLite berada di `app/riwayat_klasifikasi.db`. Aplikasi membuat tabel yang diperlukan saat dijalankan jika belum ada. Jika database ini disertakan di Git, riwayat yang sudah tersimpan di dalamnya juga akan ikut dibagikan.
