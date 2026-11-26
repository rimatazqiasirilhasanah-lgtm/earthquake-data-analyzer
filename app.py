import streamlit as st
import pandas as pd
import numpy as np
from machine_learning import tampilkan_machine_learning

# KONFIGURASI WEBSITE
st.set_page_config(
    page_title="Earthquake Data Analyzer",
    page_icon="🌏",
    layout="wide"
)

# TITIK REFERENSI BANDUNG
BANDUNG = (-6.9175, 107.6191)

# JUDUL WEBSITE
st.title("🌏 Earthquake Data Analyzer")
st.subheader(
    "Sistem Klasifikasi Tingkat Gempa Bumi "
    "Berdasarkan Parameter Seismik di Wilayah Bandung"
)

st.markdown("""
**Physics Project | Universitas Pendidikan Indonesia**

Aplikasi untuk mengolah, menganalisis, dan
mengklasifikasikan data gempa bumi berdasarkan
parameter seismik.
""")

st.divider()

# PENGATURAN ANALISIS
st.sidebar.header("Pengaturan Analisis")

radius = st.sidebar.number_input(
    "Radius Bandung (km)",
    min_value=10,
    max_value=500,
    value=150
)

b1 = st.sidebar.number_input(
    "Batas Rendah - Sedang",
    value=2.0
)

b2 = st.sidebar.number_input(
    "Batas Sedang - Kuat",
    value=3.0
)

# FUNGSI JARAK HAVERSINE
def haversine(lat, lon):
    lat1, lon1 = np.radians(BANDUNG)
    lat2 = np.radians(lat)
    lon2 = np.radians(lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * 6371 * np.arcsin(np.sqrt(a))


# FUNGSI KLASIFIKASI
def klasifikasi(m):
    if m < b1:
        return "Rendah"
    elif m < b2:
        return "Sedang"
    return "Kuat"


# UPLOAD DATASET
st.header("1. Upload Dataset Gempa")

file = st.file_uploader(
    "Pilih dataset BMKG",
    type=["csv", "xlsx", "xls"]
)

if file is not None:

    try:
        if file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        # NORMALISASI NAMA KOLOM
        df.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        aliases = {
            "waktu": [
                "date time (utc)", "time",
                "waktu", "datetime"
            ],
            "lat": ["latitude", "lat", "lintang"],
            "lon": ["longitude", "lon", "bujur"],
            "mag": ["magnitude", "mag", "magnitudo"],
            "depth": [
                "depth (km)", "depth", "kedalaman"
            ]
        }

        mapping = {}

        for standard, names in aliases.items():
            found = next(
                (c for c in df.columns if c in names),
                None
            )

            if found:
                mapping[found] = standard

        df = df.rename(columns=mapping)

        missing = [
            c for c in aliases if c not in df.columns
        ]

        if missing:
            st.error(
                "Kolom dataset belum lengkap: "
                + ", ".join(missing)
            )
            st.stop()

        # PEMBERSIHAN DATA
        for c in ["lat", "lon", "mag", "depth"]:
            df[c] = pd.to_numeric(
                df[c], errors="coerce"
            )

        df["waktu"] = pd.to_datetime(
            df["waktu"], errors="coerce", utc=True
        )

        df = df.dropna(
            subset=["waktu", "lat", "lon", "mag", "depth"]
        )

        df = df.drop_duplicates(
            subset=["waktu", "lat", "lon", "mag"]
        )

        df = df[
            df["mag"].between(0, 10)
            & (df["depth"] >= 0)
            & df["lat"].between(-90, 90)
            & df["lon"].between(-180, 180)
        ].copy()

        # HITUNG JARAK BANDUNG
        df["jarak_bdg"] = haversine(
            df["lat"].to_numpy(),
            df["lon"].to_numpy()
        )

        df = df[
            df["jarak_bdg"] <= radius
        ].copy()

        if b1 >= b2:
            st.error(
                "Batas Rendah-Sedang harus lebih kecil "
                "daripada batas Sedang-Kuat."
            )
            st.stop()

        # KLASIFIKASI MAGNITUDO
        df["kelas"] = df["mag"].apply(klasifikasi)

        # KONVERSI WAKTU UTC KE WIB
        df["wib"] = (
            df["waktu"]
            .dt.tz_convert("Asia/Jakarta")
        )

        # ENERGI GEMPA
        df["energi_J"] = (
            10 ** (1.5 * df["mag"] + 4.8)
        )

        df = df.sort_values("waktu")

        st.success("Dataset berhasil diproses!")

        # RINGKASAN
        st.header("2. Ringkasan Data")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Jumlah Gempa", len(df))

        c2.metric(
            "Magnitudo Rata-rata",
            f"{df['mag'].mean():.2f}"
        )

        c3.metric(
            "Magnitudo Maksimum",
            f"{df['mag'].max():.1f}"
        )

        c4.metric(
            "Kedalaman Rata-rata",
            f"{df['depth'].mean():.1f} km"
        )

        # TABEL DATA
        st.header("3. Hasil Klasifikasi")

        st.dataframe(
            df[
                [
                    "wib", "lat", "lon",
                    "mag", "depth",
                    "jarak_bdg", "kelas",
                    "energi_J"
                ]
            ],
            use_container_width=True
        )

        # DISTRIBUSI KELAS
        st.header("4. Distribusi Tingkat Gempa")

        distribusi = (
            df["kelas"]
            .value_counts()
            .reindex(
                ["Rendah", "Sedang", "Kuat"],
                fill_value=0
            )
        )

        st.bar_chart(distribusi)

        # VISUALISASI MAGNITUDO
        st.header("5. Visualisasi Magnitudo")

        st.line_chart(
            df.set_index("wib")["mag"]
        )

        # PETA SEBARAN
        st.header("6. Sebaran Gempa")

        st.map(
            df.rename(
                columns={"lat": "latitude", "lon": "longitude"}
            ),
            latitude="latitude",
            longitude="longitude"
        )

        # INFORMASI FISIKA
        st.header("7. Analisis Fisika")

        total_energi = df["energi_J"].sum()

        st.write(
            "Rumus energi seismik:"
        )

        st.latex(
            r"\\log_{10} E = 1.5M + 4.8"
        )

        st.metric(
            "Total Energi Seismik",
            f"{total_energi:.3e} Joule"
        )

                # MACHINE LEARNING
        tampilkan_machine_learning(df)

        # DOWNLOAD HASIL
        hasil = df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            "Download Hasil Klasifikasi",
            data=hasil,
            file_name="hasil_klasifikasi_gempa.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    st.info(
        "Silakan upload dataset gempa BMKG "
        "untuk memulai analisis."
    )

    st.markdown("""
    ### Fitur aplikasi
    - Pengolahan dataset gempa
    - Klasifikasi tingkat gempa
    - Statistik parameter seismik
    - Visualisasi data
    - Analisis energi gempa
    - Ekspor hasil klasifikasi
    """)

st.divider()

st.caption(
    "Earthquake Data Analyzer | Physics Project UPI"
)
