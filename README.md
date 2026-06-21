# 📊 Data Mining - Clustering App (K-Means)

Aplikasi web sederhana untuk **clustering (pengelompokan data)** menggunakan algoritma **K-Means**, dibangun dengan **Streamlit**.

Cocok untuk tugas kuliah / belajar Data Mining.

## ✨ Fitur
- Pakai dataset contoh (Iris) atau upload CSV sendiri
- Pemilihan kolom/fitur secara interaktif
- Elbow Method untuk menentukan jumlah cluster terbaik
- Visualisasi hasil cluster (otomatis pakai PCA jika fitur > 2)
- Silhouette Score untuk evaluasi kualitas cluster
- Download hasil clustering sebagai CSV

## 🗂️ Struktur Folder
```
clustering-app/
├── app.py              # Kode utama aplikasi Streamlit
├── requirements.txt    # Daftar library yang dibutuhkan
├── .gitignore
└── README.md
```

## 🚀 Cara Menjalankan di Lokal
1. Pastikan Python sudah terinstall (versi 3.9+)
2. Buat virtual environment (opsional tapi disarankan):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install semua library:
   ```
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```
5. Browser otomatis terbuka di `http://localhost:8501`

## ☁️ Cara Deploy ke Streamlit Community Cloud
1. Push project ini ke GitHub (lihat tutorial lengkap di chat)
2. Buka https://share.streamlit.io
3. Login dengan akun GitHub
4. Klik **New app**, pilih repository ini
5. Pilih file utama: `app.py`
6. Klik **Deploy**

## 🧠 Tentang Algoritma
**K-Means Clustering** adalah algoritma unsupervised learning yang mengelompokkan data
ke dalam *K* kelompok berdasarkan jarak antar titik data. Algoritma ini cocok digunakan
untuk segmentasi pelanggan, pengelompokan jenis bunga, pola pembelian, dan lain-lain.

## 📄 Lisensi
Bebas digunakan untuk keperluan belajar/tugas.
