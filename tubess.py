import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# PATH BASE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. LOAD DATA + NORMALISASI

def read_locations(path_xlsx="locations.xlsx", path_csv="locations.csv"):
    if os.path.exists(os.path.join(BASE_DIR, path_xlsx)):
        df = pd.read_excel(os.path.join(BASE_DIR, path_xlsx))
    elif os.path.exists(os.path.join(BASE_DIR, path_csv)):
        df = pd.read_csv(os.path.join(BASE_DIR, path_csv))
    else:
        st.error("File locations.xlsx atau locations.csv tidak ditemukan.")
        st.stop()
    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        "nama": "name",
        "harga_per_m2": "price_per_m2",
        "resiko_banjir": "flood_risk",
        "tingkat_keramaian": "crowd_level",
        "persentase_rth": "rth_percent",
        "lokasi_strategis": "proximity_public"
    }
    df = df.rename(columns=rename_map)
    required = ["name", "price_per_m2", "flood_risk", "crowd_level", "rth_percent", "proximity_public"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"Kolom berikut hilang di file Excel/CSV: {missing}")
        st.stop()

    ind_to_en = {
        "rendah": "low", "sedang": "medium", "tinggi": "high",
        "low": "low", "medium": "medium", "high": "high"
    }
    def map_ind(val):
        if pd.isna(val):
            return "medium"
        v = str(val).strip().lower()
        return ind_to_en.get(v, "medium")

    df["flood_risk"] = df["flood_risk"].apply(map_ind)
    df["crowd_level"] = df["crowd_level"].apply(map_ind)
    df["proximity_public"] = df["proximity_public"].apply(map_ind)
    df["price_per_m2_million"] = pd.to_numeric(df["price_per_m2"], errors="coerce").fillna(0)
    df["price_per_m2"] = df["price_per_m2_million"] * 1_000_000
    df["rth_percent"] = pd.to_numeric(df["rth_percent"], errors="coerce").fillna(0)

    return df

# 2. SCORING

FLOOD_MAP = {"low": 1.0, "medium": 0.5, "high": 0.0}
CROWD_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
PROX_MAP  = {"low": 0.0, "medium": 0.5, "high": 1.0}

def normalize_price_scores(df):
    prices = df["price_per_m2"].values.astype(float)
    mn, mx = prices.min(), prices.max()
    if mn == mx:
        return np.ones_like(prices)
    return (mx - prices) / (mx - mn)

def compute_scores(df):
    price_scores = normalize_price_scores(df)
    flood_scores = df["flood_risk"].map(FLOOD_MAP).fillna(0.5).values
    crowd_scores = df["crowd_level"].map(CROWD_MAP).fillna(0.5).values
    prox_scores  = df["proximity_public"].map(PROX_MAP).fillna(0.5).values

    r = df["rth_percent"].values.astype(float)
    if r.max() != r.min():
        rth_scores = (r - r.min()) / (r.max() - r.min())
    else:
        rth_scores = np.ones_like(r)

    weights = {
        "price": 0.40, "flood": 0.30, "crowd": 0.15,
        "prox":  0.10, "rth":   0.05
    }

    final = (
        weights["price"] * price_scores +
        weights["flood"] * flood_scores +
        weights["crowd"] * crowd_scores +
        weights["prox"]  * prox_scores +
        weights["rth"]   * rth_scores
    )

    df2 = df.copy()
    df2["score"] = final
    df2["price_score"] = price_scores
    df2["flood_score"] = flood_scores
    df2["crowd_score"] = crowd_scores
    df2["prox_score"] = prox_scores
    df2["rth_score"] = rth_scores

    return df2.sort_values("score", ascending=False)

# UTIL: sanitize & flexible image path

def sanitize_filename(name):
    
    s = name.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s + ".jpg"

def possible_image_paths(name):
    """Return list of candidate image paths to try (most likely first)."""
    candidates = []
    candidates.append(os.path.join(BASE_DIR, sanitize_filename(name)))
    candidates.append(os.path.join(BASE_DIR, name.lower().replace(" ", "") + ".jpg"))
    candidates.append(os.path.join(BASE_DIR, name.lower().replace(" ", "-") + ".jpg"))
    candidates.append(os.path.join(BASE_DIR, name + ".jpg"))
    dirpath = os.path.join(BASE_DIR, "static", "images", "lokasi")
    for p in list(candidates):
        candidates.append(os.path.join(dirpath, os.path.basename(p)))    
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique

def find_existing_image(name):
    for p in possible_image_paths(name):
        if os.path.exists(p):
            return p
    return None

# 3. STREAMLIT UI

st.set_page_config(layout="wide")
st.title("🏡 Rekomendasi Pembelian Tanah di Kota Bandung")
st.markdown("""
<div style="
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    font-size: 20px;
    color: #444;
">
    <marquee behavior="scroll" direction="left" scrollamount="9">
        Masukkan budget anda dan luas tanah untuk menampilkan rekomendasi terbaik. Terima Kasih 😊😊😊
    </marquee>
</div>
""", unsafe_allow_html=True)

budget_miliar = st.number_input("💰 Masukkan Budget (dalam MILIAR)", min_value=0.1, step=0.1, value=1.0)
budget = budget_miliar * 1_000_000_000
luas = st.number_input("📏 Masukkan Luas Tanah (m²)", min_value=20, step=5, value=100)
df = read_locations()
max_loc = len(df)
jumlah_rekom = st.number_input("🔢 Jumlah lokasi yang ingin dianalisis:", min_value=1, max_value=max_loc, value=min(10, max_loc), step=1)

# Tampilkan bobot ke user sebelum tombol

st.markdown("### 📘 Bobot Penilaian Lokasi")
st.markdown("""
- **Harga tanah:** 40% (semakin murah → skor lebih tinggi)  
- **Risiko banjir:** 30% (rendah → skor lebih tinggi)  
- **Tingkat keramaian:** 15% (low → skor lebih tinggi)  
- **Akses fasilitas publik:** 10% (high → skor lebih tinggi)  
- **RTH:** 5% (persentase RTH lebih besar → skor lebih tinggi)  
""")
st.caption("Penjelasan: skor akhir dihitung dengan menggabungkan kelima kriteria di atas sesuai bobot. Grafik menampilkan skor akhir dalam persen (0–100%).")

# KETIKA TOMBOL DIKLIK

if st.button("Tampilkan Rekomendasi"):
    scored = compute_scores(df)
    scored["total_price"] = scored["price_per_m2"] * luas
    affordable = scored[scored["total_price"] <= budget]
    if affordable.empty:
        st.error("Tidak ada lokasi sesuai budget Anda.")
        st.stop()
    topk = affordable.head(jumlah_rekom).reset_index(drop=True)
    top3 = topk.head(3).reset_index(drop=True)

    def format_total_price(x):
        juta = x / 1_000_000
        if juta < 1000:
            return f"{juta:.0f} juta"
        return f"{juta/1000:.1f} miliar"

    display_tbl = topk[[
        "name", "price_per_m2_million", "total_price",
        "flood_risk", "crowd_level", "proximity_public",
        "rth_percent"
    ]].copy()

    display_tbl = display_tbl.rename(columns={
        "name": "Nama",
        "price_per_m2_million": "Harga_per_m2 (juta/m²)",
        "total_price": "Harga_total",
        "flood_risk": "Risiko_Banjir",
        "crowd_level": "Tingkat_Keramaian",
        "proximity_public": "Lokasi_Strategis",
        "rth_percent": "RTH (%)",
    })

    display_tbl["Harga_total"] = display_tbl["Harga_total"].apply(format_total_price)
    display_tbl["Harga_per_m2 (juta/m²)"] = display_tbl["Harga_per_m2 (juta/m²)"].apply(lambda x: f"{x:.0f} juta/m²")
    display_tbl["RTH (%)"] = display_tbl["RTH (%)"].apply(lambda x: f"{x:.0f}%")

    st.subheader(f"📌 {len(topk)} Lokasi yang Dianalisis (sesuai budget)")
    st.dataframe(display_tbl, use_container_width=True)

    # 4. REKOMENDASI TOP 3
    
    st.subheader("🏆 3 Rekomendasi Lokasi Terbaik")

    informasi_lokasi = {
        "arcamanik": {
            "kelebihan": ["Akses jalan mudah", "Harga relatif terjangkau"],
            "kekurangan": ["Beberapa titik rawan banjir saat hujan lebat"]
        },
        "rancasari": {
            "kelebihan": ["Lingkungan tenang", "Dekat fasilitas umum"],
            "kekurangan": ["Harga sedikit lebih tinggi"]
        },
        "panyileukan": {
            "kelebihan": ["Lingkungan relatif tenang", "Harga kompetitif"],
            "kekurangan": ["Akses ke pusat kota sedikit lebih jauh"]
        },
        "mandalajati": {
            "kelebihan": ["Akses transportasi baik", "Banyak fasilitas sekitar"],
            "kekurangan": ["Area padat pada jam sibuk"]
        },
        "cidadap": {
            "kelebihan": ["Udara sejuk, lingkungan hijau", "RTH luas"],
            "kekurangan": ["Beberapa area aksesnya tidak terlalu cepat"]
        }
    }

    for i, row in top3.iterrows():
        st.markdown(f"### 📍 {row['name']}")
        img_path = find_existing_image(row["name"])
        if img_path:
            st.image(img_path, width=300)
        else:
            if row["name"].strip().lower() == "cidadap":
                st.markdown("""
                **📘 Deskripsi Lokasi Cidadap (Foto tidak tersedia)**  
                - 60% wilayah berupa dataran datar hingga berombak  
                - Ketinggian sekitar 750 mdpl  
                - Suhu harian 19°C – 28°C    
                """)
            else:
                st.info("Foto lokasi belum tersedia.")

        # --- DETAIL INFORMASI ---
        st.write(f"- **Harga per m²:** {row['price_per_m2_million']:.0f} juta/m²")
        st.write(f"- **Harga total:** {format_total_price(row['total_price'])}")
        st.write(f"- **Risiko banjir:** {row['flood_risk']}")
        st.write(f"- **Tingkat keramaian:** {row['crowd_level']}")
        st.write(f"- **Akses fasilitas publik:** {row['proximity_public']}")
        st.write(f"- **RTH:** {row['rth_percent']:.0f}%")

        # KELEBIHAN & KEKURANGAN 
        
        advantages = []
        disadvantages = []

        if row["flood_risk"] == "low":
            advantages.append("Area ini memiliki risiko banjir yang rendah.")
        else:
            disadvantages.append("Berpotensi terdampak banjir.")

        if row["crowd_level"] == "low":
            advantages.append("Lingkungan sekitar tenang.")
        elif row["crowd_level"] == "high":
            disadvantages.append("Keramaian area sekitar tinggi — kurang nyaman.")

        if row["proximity_public"] == "high":
            advantages.append("Dekat dengan fasilitas umum.")
        elif row["proximity_public"] == "low":
            disadvantages.append("Akses fasilitas umum terbatas.")

        if row["rth_percent"] >= 25:
            advantages.append("RTH luas dan memadai.")
        elif row["rth_percent"] < 15:
            disadvantages.append("RTH rendah — potensi area padat.")

        if row.get("price_score", 0) > 0.6:
            advantages.append("Harga relatif murah dibanding kecamatan lain.")
        else:
            disadvantages.append("Harga cenderung mahal.")
        key = row["name"].strip().lower().replace(" ", "").replace("-", "")
        if key in informasi_lokasi:
            info = informasi_lokasi[key]
            for x in info.get("kelebihan", []):
                if x not in advantages:
                    advantages.append(x)
            for x in info.get("kekurangan", []):
                if x not in disadvantages:
                    disadvantages.append(x)

        st.markdown("### 🟢 Kelebihan:")
        if advantages:
            for a in advantages:
                st.markdown(f"- {a}")
        else:
            st.markdown("- (Tidak ada catatan kelebihan spesifik.)")

        st.markdown("### 🔴 Kekurangan:")
        if disadvantages:
            for d in disadvantages:
                st.markdown(f"- {d}")
        else:
            st.markdown("- (Tidak ada catatan kekurangan spesifik.)")

        st.markdown("---")

    # 5. GRAFIK BAR — Perbandingan Skor Total
   
    st.subheader("📊 Perbandingan Antar Kecamatan (Top 3)")

    labels = top3["name"].tolist()
    scores_pct = (top3["score"].values * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, scores_pct, width=0.5)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Skor (%)")
    ax.set_title("Skor Total Lokasi (dalam %) — Semakin tinggi semakin direkomendasikan")
    for bar, h in zip(bars, scores_pct):
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center")

    st.pyplot(fig)
    st.markdown("""
**Keterangan skor:** Skor akhir dihitung dari kombinasi kriteria berikut dengan bobot:
- **Harga tanah:** 40%
- **Risiko banjir:** 30%
- **Tingkat keramaian:** 15%
- **Akses fasilitas publik:** 10%
- **RTH:** 5%
""")
    st.caption("Contoh interpretasi: Nilai 78% artinya lokasi memperoleh skor total 0.78 berdasarkan bobot di atas.")


# 6. RADAR CHART (SPIDER CHART)

    st.subheader("Radar Chart Perbandingan Kriteria (Top 3)")

    categories = ["Harga Lahan", "Risiko Banjir", "Tingkat Keramaian", "Akses Publik", "RTH (%)"]

    values = []
    for _, row in top3.iterrows():
        flood_risk_score = 1 - row["flood_score"]

        values.append([
            row["price_score"],
            flood_risk_score,
            row["crowd_score"],
            row["prox_score"],
            row["rth_score"]
        ])

    num_vars = len(categories)

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = plt.subplot(111, polar=True)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for i, loc in enumerate(top3["name"]):
        v = values[i]
        v = v + v[:1]   
        ax.plot(angles, v, linewidth=2, label=loc)
        ax.fill(angles, v, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=5)
    ax.set_yticks([])
    ax.set_ylim(0, 1)

    plt.title("Perbandingan Kriteria Lokasi (Radar Chart)", size=7, pad=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=4)

    st.pyplot(fig)






