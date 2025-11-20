import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
# 1. KONFIGURASI APLIKASI
# ============================
st.set_page_config(page_title="Rekomendasi Lokasi Bandung", layout="wide")

BASE_DIR = "images"  # folder gambar

st.title("📍 Sistem Rekomendasi Lokasi Terbaik di Bandung")
st.write("Menggunakan metode pembobotan variabel & visualisasi data.")


# ============================
# 2. DATASET
# ============================
data = {
    "name": ["Panyileukan", "Cidadap", "Antapani", "Kiaracondong", "Coblong"],
    "akses": [80, 75, 65, 60, 90],
    "keamanan": [70, 85, 60, 55, 80],
    "fasilitas": [78, 70, 72, 68, 88],
    "lingkungan": [82, 74, 65, 60, 85],
}

df = pd.DataFrame(data)

# Bobot variabel
weights = {
    "akses": 0.30,
    "keamanan": 0.25,
    "fasilitas": 0.25,
    "lingkungan": 0.20
}

# Info bobot untuk user
with st.expander("ℹ️ Informasi Bobot Variabel"):
    st.write("""
    **Bobot penilaian lokasi (dalam persen):**
    - Aksesibilitas: **30%**
    - Keamanan: **25%**
    - Fasilitas Umum: **25%**
    - Lingkungan & Kenyamanan: **20%**  
    """)

# Hitung skor total
df["skor"] = (
    df["akses"] * weights["akses"]
    + df["keamanan"] * weights["keamanan"]
    + df["fasilitas"] * weights["fasilitas"]
    + df["lingkungan"] * weights["lingkungan"]
)

# ============================
# 3. TOP 3 REKOMENDASI
# ============================
st.subheader("🏆 3 Rekomendasi Lokasi Terbaik")

top3 = df.sort_values("skor", ascending=False).head(3)


# Database kelebihan & kekurangan
insight = {
    "Panyileukan": {
        "plus": ["Akses cukup baik", "Lingkungan cukup nyaman", "Fasilitas memadai"],
        "minus": ["Keamanan tidak terlalu tinggi dibanding Coblong/Cidadap"],
    },
    "Cidadap": {
        "plus": ["Keamanan sangat bagus", "Udara sejuk & nyaman"],
        "minus": ["Aksesibilitas tidak sebaik Coblong", "Fasilitas tidak sebanyak pusat kota"],
    },
    "Antapani": {
        "plus": ["Fasilitas umum cukup lengkap"],
        "minus": ["Akses & keamanan sedang"],
    },
    "Kiaracondong": {
        "plus": ["Akses transportasi cukup banyak"],
        "minus": ["Keamanan rendah", "Lingkungan padat"],
    },
    "Coblong": {
        "plus": ["Akses terbaik", "Fasilitas sangat lengkap", "Lingkungan nyaman"],
        "minus": ["Area ramai dan cukup macet"],
    },
}

for i, row in top3.iterrows():

    st.markdown(f"### 📍 {row['name']} (Skor: **{row['skor']:.2f}%**)")

    # Nama file gambar
    img_file = row["name"].lower().replace(" ", "") + ".jpg"
    img_path = os.path.join(BASE_DIR, img_file)

    # Tampilkan gambar
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("Foto lokasi belum tersedia.")

    # Tampilkan plus minus
    st.write("#### ✔️ Kelebihan")
    for p in insight[row["name"]]["plus"]:
        st.write(f"- {p}")

    st.write("#### ⚠️ Kekurangan")
    for m in insight[row["name"]]["minus"]:
        st.write(f"- {m}")

    st.markdown("---")


# ============================
# 4. GRAFIK BATANG (DENGAN TEKS PENJELAS)
# ============================
st.subheader("📊 Perbandingan Skor Lokasi")

fig, ax = plt.subplots()
bars = ax.bar(df["name"], df["skor"])

# Tambahkan teks penjelas di atas bar
for i, b in enumerate(bars):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        f"{df['skor'][i]:.1f}%",
        ha="center",
        va="bottom"
    )

st.pyplot(fig)


# ============================
# 5. SPIDER CHART
# ============================
st.subheader("🕸️ Radar Chart Per Lokasi (Spider Chart Berbahasa Indonesia)")

selected = st.selectbox("Pilih lokasi:", df["name"])

categories = ["Akses", "Keamanan", "Fasilitas", "Lingkungan"]

values = df[df["name"] == selected][["akses", "keamanan", "fasilitas", "lingkungan"]].values.flatten()
values = np.append(values, values[0])  # ulangi nilai pertama untuk menutup grafik

angles = np.linspace(0, 2 * np.pi, len(categories) + 1)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

st.pyplot(fig)
