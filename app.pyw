import streamlit as st
import pandas as pd
import joblib


# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Segmentasi Profil Fisik & Pola Hidup",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return kmeans, scaler


@st.cache_data
def load_profile():
    return pd.read_csv("cluster_profile.csv")


kmeans, scaler = load_model()
cluster_profile = load_profile()


# ==========================================
# JUDUL
# ==========================================

st.title("📊 Segmentasi Profil Fisik & Pola Hidup")

st.write(
    "Aplikasi ini menggunakan algoritma K-Means Clustering "
    "untuk mengelompokkan individu berdasarkan karakteristik "
    "fisik dan pola hidup."
)

st.divider()


# ==========================================
# INPUT DATA
# ==========================================

st.subheader("Masukkan Data Individu")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Usia",
        min_value=14.0,
        max_value=61.0,
        value=22.0,
        step=0.1
    )

    height = st.number_input(
        "Tinggi Badan (meter)",
        min_value=1.45,
        max_value=1.98,
        value=1.70,
        step=0.01
    )

    weight = st.number_input(
        "Berat Badan (kg)",
        min_value=39.0,
        max_value=173.0,
        value=70.0,
        step=0.1
    )

    fcvc = st.number_input(
        "Frekuensi Konsumsi Sayur (FCVC)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

with col2:

    ncp = st.number_input(
        "Jumlah Makan Utama (NCP)",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    ch2o = st.number_input(
        "Konsumsi Air (CH2O)",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    faf = st.number_input(
        "Aktivitas Fisik (FAF)",
        min_value=0.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    tue = st.number_input(
        "Penggunaan Teknologi (TUE)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )


# ==========================================
# ANALISIS
# ==========================================

if st.button("🔍 Analisis Profil", use_container_width=True):

    # Membuat DataFrame dari input
    data_baru = pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CH2O": [ch2o],
        "FAF": [faf],
        "TUE": [tue]
    })

    # Standardisasi menggunakan scaler training
    data_scaled = scaler.transform(data_baru)

    data_scaled = pd.DataFrame(
        data_scaled,
        columns=data_baru.columns
    )

    # Menentukan cluster
    cluster = int(kmeans.predict(data_scaled)[0])

    st.divider()

    # ======================================
    # HASIL
    # ======================================

    st.subheader("Hasil Segmentasi")

    st.success(
        f"Individu termasuk ke dalam **Cluster {cluster}**"
    )

    # Mengambil profil cluster
    profile = cluster_profile[
        cluster_profile["Cluster"] == cluster
    ].iloc[0]

    # ======================================
    # INFORMASI SINGKAT
    # ======================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Anggota",
            int(profile["Jumlah Data"])
        )

    with col2:
        st.metric(
            "Rata-rata Berat",
            f"{profile['Weight']:.2f} kg"
        )

    with col3:
        st.metric(
            "Rata-rata Aktivitas Fisik",
            f"{profile['FAF']:.2f}"
        )

    # ======================================
    # PROFIL CLUSTER
    # ======================================

    st.subheader("Profil Cluster")

    profile_display = profile[
        [
            "Age",
            "Height",
            "Weight",
            "FCVC",
            "NCP",
            "CH2O",
            "FAF",
            "TUE"
        ]
    ].to_frame("Rata-rata")

    st.dataframe(
        profile_display.round(2),
        use_container_width=True
    )

    # ======================================
    # INFORMASI TAMBAHAN
    # ======================================

    st.info(
        f"Kategori NObeyesdad yang paling banyak ditemukan pada cluster ini "
        f"adalah **{profile['Kategori Dominan']}** "
        f"({profile['Persentase Dominan (%)']:.2f}%)."
    )