import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA + NORMALISASI
# ============================================================

def read_locations(path_xlsx="locations.xlsx", path_csv="locations.csv"):

    if os.path.exists(path_xlsx):
        df = pd.read_excel(path_xlsx)
    elif os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
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
        st.error(f"Kolom berikut hilang di file Excel: {missing}")
        st.stop()

    ind_to_en = {
        "rendah": "low", "sedang": "medium", "tinggi": "high",
        "low": "low", "medium": "medium", "high": "high"
    }

    def map_ind(val):
        if pd.isna(val): return "medium"
        v = str(val).strip().lower()
        return ind_to_en.get(v, "medium")

    df["flood_risk"] = df["flood_risk"].apply(map_ind)
    df["crowd_level"] = df["crowd_level"].apply(map_ind)
    df["proximity_public"] = df["proximity_public"].apply(map_ind)

    df["price_per_m2_million"] = pd.to_numeric(df["price_per_m2"], errors="coerce").fillna(0)
    df["price_per_m2"] = df["price_per_m2_million"] * 1_000_000

    df["rth_percent"] = pd.to_numeric(df["rth_percent"], errors="coerce").fillna(0)

    return df


# ============================================================
# 2. SCORING
# ============================================================

FLOOD_MAP = {"low": 1.0, "medium": 0.5, "high": 0.0}
CROWD_MAP = {"low": 1.0, "medium": 0.5, "high": 0.0}
PROX_MAP  = {"low": 0.0, "medium": 0.5, "high": 1.0}

def normalize_price_scores(df):
    prices = df["price_per_m2"].values
    mn, mx = prices.min(), prices.max()
    if mn == mx:
        return np.ones_like(prices)
    return (mx - prices) / (mx - mn)

def compute_scores(df):
    price_scores = normalize_price_scores(df)
    flood_scores = df["flood_risk"].map(FLOOD_MAP).fillna(0.5)
    crowd_scores = df["crowd_level"].map(CROWD_MAP).fillna(0.5)
    prox_scores  = df["proximity_public"].map(PROX_MAP).fillna(0.5)

    r = df["rth_percent"].values
    rth_scores = (r - r.min()) / (r.max() - r.min()) if r.max() != r.min() else np.ones_like(r)

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


# ============================================================
# 3. STREAMLIT UI
# ============================================================

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



budget_miliar = st.number_input("💰 Masukkan Budget (dalam MILIAR)", min_value=0.1, step=0.1)
budget = budget_miliar * 1_000_000_000

luas = st.number_input("📏 Masukkan Luas Tanah (m²)", min_value=20, step=5)

df = read_locations()
max_loc = len(df)

jumlah_rekom = st.number_input("🔢 Jumlah lokasi yang ingin dianalisis:", 3, max_loc, min(10, max_loc))

# ============================================================
# KETIKA TOMBOL DIKLIK
# ============================================================

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

    # ============================================================
    # 4. REKOMENDASI TOP 3
    # ============================================================

   st.subheader("🏆 3 Rekomendasi Lokasi Terbaik")

for i, row in top3.iterrows():
    st.markdown(f"### 📍 {row['name']}")

    # --- GENERATE PATH GAMBAR ---
    img_file = row["name"].lower().replace(" ", "") + ".jpg"
    img_path = os.path.join(BASE_DIR, img_file)

    # --- TAMPILKAN GAMBAR ---
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)
    else:
        if row["name"].lower() == "cidadap":
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

        # =====================================================
        # KELEBIHAN & KEKURANGAN
        # =====================================================

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

        if row["price_score"] > 0.6:
            advantages.append("Harga relatif murah dibanding kecamatan lain.")
        else:
            disadvantages.append("Harga cenderung mahal.")

        st.markdown("### 🟢 Kelebihan:")
        for a in advantages:
            st.markdown(f"- {a}")

        st.markdown("### 🔴 Kekurangan:")
        for d in disadvantages:
            st.markdown(f"- {d}")

        st.markdown("---")

    # ============================================================
    # 5. GRAFIK
    # ============================================================

    st.subheader("📊 Perbandingan Antar Kecamatan (Top 3)")

    labels = top3["name"].tolist()
    scores_pct = (top3["score"].values * 100).round(1)

    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, scores_pct, color=colors[:len(labels)], width=0.5)

    ax.set_ylim(0, 100)
    ax.set_ylabel("(%)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center"
        )

    st.pyplot(fig)

