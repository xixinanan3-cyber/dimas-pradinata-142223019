"""
APLIKASI DATA MINING - CLUSTERING (PENGELOMPOKAN DATA)
========================================================
Aplikasi ini menggunakan algoritma K-Means untuk mengelompokkan data
ke dalam beberapa cluster/kelompok berdasarkan kemiripan ciri-cirinya.

Dataset default: Iris (bunga) - bisa diganti dengan upload CSV sendiri.

Cara jalankan di lokal:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Data Mining - Clustering App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Aplikasi Data Mining: Clustering dengan K-Means")
st.markdown("""
Aplikasi ini melakukan **clustering (pengelompokan data)** menggunakan algoritma **K-Means**.
Kamu bisa memakai dataset bawaan (Iris) atau upload file CSV sendiri.
""")

# ---------------------------------------------------------
# 2. SIDEBAR - PILIHAN SUMBER DATA
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")

sumber_data = st.sidebar.radio(
    "Pilih sumber data:",
    ["Dataset Contoh (Iris)", "Upload CSV Sendiri"]
)

if sumber_data == "Dataset Contoh (Iris)":
    iris = load_iris(as_frame=True)
    df = iris.frame.drop(columns=["target"])  # hilangkan label asli, supaya unsupervised murni
    st.sidebar.success("Menggunakan dataset Iris bawaan (150 baris).")
else:
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File berhasil diupload: {uploaded_file.name}")
    else:
        st.warning("⬅️ Silakan upload file CSV di sidebar untuk melanjutkan.")
        st.stop()

# ---------------------------------------------------------
# 3. TAMPILKAN DATA MENTAH
# ---------------------------------------------------------
st.subheader("1️⃣ Data Mentah")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Jumlah baris: {df.shape[0]} | Jumlah kolom: {df.shape[1]}")

# Ambil hanya kolom numerik (K-Means hanya bisa baca angka)
kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()

if len(kolom_numerik) < 2:
    st.error("Dataset harus punya minimal 2 kolom numerik untuk clustering.")
    st.stop()

# ---------------------------------------------------------
# 4. PILIH FITUR UNTUK CLUSTERING
# ---------------------------------------------------------
st.subheader("2️⃣ Pilih Fitur (Kolom) untuk Clustering")
fitur_terpilih = st.multiselect(
    "Pilih minimal 2 kolom numerik:",
    options=kolom_numerik,
    default=kolom_numerik[:min(4, len(kolom_numerik))]
)

if len(fitur_terpilih) < 2:
    st.warning("Pilih minimal 2 kolom untuk melanjutkan.")
    st.stop()

X = df[fitur_terpilih].dropna()

# Standarisasi data (penting! supaya skala antar kolom seimbang)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 5. ELBOW METHOD - BANTU MENENTUKAN JUMLAH CLUSTER OPTIMAL
# ---------------------------------------------------------
st.subheader("3️⃣ Elbow Method (Menentukan Jumlah Cluster Optimal)")

with st.expander("ℹ️ Apa itu Elbow Method?"):
    st.write("""
    Elbow Method membantu kita menebak jumlah cluster (K) terbaik.
    Caranya: lihat grafik di bawah, cari titik di mana garis mulai
    melandai membentuk siku (elbow). Titik itu biasanya jumlah K yang ideal.
    """)

max_k = min(10, len(X) - 1)
inertia_list = []
k_range = range(1, max_k + 1)

for k in k_range:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_scaled)
    inertia_list.append(km_temp.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(list(k_range), inertia_list, marker="o")
ax_elbow.set_xlabel("Jumlah Cluster (K)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Elbow Method")
st.pyplot(fig_elbow)

# ---------------------------------------------------------
# 6. PILIH JUMLAH CLUSTER & JALANKAN K-MEANS
# ---------------------------------------------------------
st.subheader("4️⃣ Jalankan Clustering")

jumlah_cluster = st.slider(
    "Pilih jumlah cluster (K):",
    min_value=2, max_value=max_k, value=3
)

kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
label_cluster = kmeans.fit_predict(X_scaled)

df_hasil = df.loc[X.index].copy()
df_hasil["Cluster"] = label_cluster

# Hitung silhouette score (semakin mendekati 1, semakin bagus pengelompokannya)
sil_score = silhouette_score(X_scaled, label_cluster)
st.metric("Silhouette Score", f"{sil_score:.3f}",
          help="Rentang -1 sampai 1. Semakin tinggi, semakin bagus kualitas cluster.")

# ---------------------------------------------------------
# 7. VISUALISASI HASIL CLUSTER (2D, pakai PCA jika fitur > 2)
# ---------------------------------------------------------
st.subheader("5️⃣ Visualisasi Hasil Cluster")

if len(fitur_terpilih) > 2:
    pca = PCA(n_components=2)
    komponen = pca.fit_transform(X_scaled)
    df_plot = pd.DataFrame(komponen, columns=["PC1", "PC2"])
    df_plot["Cluster"] = label_cluster
    x_axis, y_axis = "PC1", "PC2"
    st.caption("Karena fitur > 2, data diproyeksikan ke 2 dimensi menggunakan PCA.")
else:
    df_plot = X.copy()
    df_plot["Cluster"] = label_cluster
    x_axis, y_axis = fitur_terpilih[0], fitur_terpilih[1]

fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df_plot, x=x_axis, y=y_axis,
    hue="Cluster", palette="tab10", s=80, ax=ax_scatter
)
ax_scatter.set_title("Hasil Pengelompokan Data")
st.pyplot(fig_scatter)

# ---------------------------------------------------------
# 8. TABEL HASIL + DOWNLOAD
# ---------------------------------------------------------
st.subheader("6️⃣ Tabel Hasil Clustering")
st.dataframe(df_hasil, use_container_width=True)

csv_hasil = df_hasil.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Hasil sebagai CSV",
    data=csv_hasil,
    file_name="hasil_clustering.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# 9. RINGKASAN PER CLUSTER
# ---------------------------------------------------------
st.subheader("7️⃣ Ringkasan Rata-Rata per Cluster")
ringkasan = df_hasil.groupby("Cluster")[fitur_terpilih].mean()
st.dataframe(ringkasan, use_container_width=True)

st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit & Scikit-learn untuk tugas Data Mining.")
