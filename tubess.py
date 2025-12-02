import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONSTANTS ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Scoring weights
WEIGHTS = {
    "price": 0.40,
    "flood": 0.30,
    "crowd": 0.15,
    "prox": 0.10,
    "rth": 0.05
}

# Mapping constants
FLOOD_MAP = {"low": 1.0, "medium": 0.5, "high": 0.0}
CROWD_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
PROX_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}

# RTH thresholds
RTH_THRESHOLD_HIGH = 25
RTH_THRESHOLD_LOW = 15

# Price score threshold for advantages
PRICE_SCORE_THRESHOLD = 0.6

# Budget limits
MIN_BUDGET_MILIAR = 0.1
MAX_BUDGET_MILIAR = 1000.0
MIN_LUAS = 20

IND_TO_EN = {
    "rendah": "low",
    "sedang": "medium",
    "tinggi": "high",
    "low": "low",
    "medium": "medium",
    "high": "high"
}

# ============ LOCATION INFORMATION ============
LOCATION_INFO: Dict[str, Dict[str, List[str]]] = {
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

# ============ DATA LOADING ============

@st.cache_data
def read_locations(path_xlsx: str = "locations.xlsx", path_csv: str = "locations.csv") -> pd.DataFrame:
    """
    Membaca dan memproses data lokasi dari file Excel atau CSV.
    
    Args:
        path_xlsx: Path ke file Excel
        path_csv: Path ke file CSV
        
    Returns:
        DataFrame yang sudah diproses
    """
    try:
        # Coba baca Excel dulu
        xlsx_path = os.path.join(BASE_DIR, path_xlsx)
        csv_path = os.path.join(BASE_DIR, path_csv)
        
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            logger.info(f"✅ Berhasil membaca file: {path_xlsx}")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Berhasil membaca file: {path_csv}")
        else:
            error_msg = f"""
            ❌ File tidak ditemukan!
            
            Pastikan salah satu file berikut ada di folder:
            📁 {BASE_DIR}
            
            File yang dicari:
            - {path_xlsx} ATAU
            - {path_csv}
            
            Silakan upload file terlebih dahulu.
            """
            st.error(error_msg)
            logger.error(f"File tidak ditemukan di {BASE_DIR}")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {str(e)}")
        logger.error(f"Error reading file: {e}")
        st.stop()
    
    # Normalisasi nama kolom
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Mapping nama kolom
    rename_map = {
        "nama": "name",
        "harga_per_m2": "price_per_m2",
        "resiko_banjir": "flood_risk",
        "tingkat_keramaian": "crowd_level",
        "persentase_rth": "rth_percent",
        "lokasi_strategis": "proximity_public"
    }
    df = df.rename(columns=rename_map)
    
    # Validasi kolom yang diperlukan
    required = ["name", "price_per_m2", "flood_risk", "crowd_level", "rth_percent", "proximity_public"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"❌ Kolom berikut hilang di file: {', '.join(missing)}")
        logger.error(f"Missing columns: {missing}")
        st.stop()
    
    # Map kategori
    def map_category(val) -> str:
        """Map kategori dengan safe default"""
        if pd.isna(val):
            return "medium"
        v = str(val).strip().lower()
        return IND_TO_EN.get(v, "medium")
    
    df["flood_risk"] = df["flood_risk"].apply(map_category)
    df["crowd_level"] = df["crowd_level"].apply(map_category)
    df["proximity_public"] = df["proximity_public"].apply(map_category)
    
    # Convert harga dan RTH
    df["price_per_m2_million"] = pd.to_numeric(df["price_per_m2"], errors="coerce").fillna(0)
    df["price_per_m2"] = df["price_per_m2_million"] * 1_000_000
    df["rth_percent"] = pd.to_numeric(df["rth_percent"], errors="coerce").fillna(0)
    
    logger.info(f"✅ Berhasil memproses {len(df)} lokasi")
    return df

# ============ SCORING FUNCTIONS ============

def normalize_price_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Normalisasi skor harga: semakin murah semakin tinggi skornya.
    
    Args:
        df: DataFrame dengan kolom price_per_m2
        
    Returns:
        Array skor yang dinormalisasi (0-1)
    """
    prices = df["price_per_m2"].values.astype(float)
    mn, mx = prices.min(), prices.max()
    
    if mn == mx:
        logger.warning("Semua harga sama, returning uniform scores")
        return np.ones_like(prices)
    
    return (mx - prices) / (mx - mn)

@st.cache_data
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung skor total untuk setiap lokasi berdasarkan kriteria.
    
    Args:
        df: DataFrame dengan data lokasi
        
    Returns:
        DataFrame dengan kolom skor tambahan
    """
    # Hitung skor individual
    price_scores = normalize_price_scores(df)
    flood_scores = df["flood_risk"].map(FLOOD_MAP).fillna(0.5).values
    crowd_scores = df["crowd_level"].map(CROWD_MAP).fillna(0.5).values
    prox_scores = df["proximity_public"].map(PROX_MAP).fillna(0.5).values
    
    # Normalisasi RTH
    r = df["rth_percent"].values.astype(float)
    if r.max() != r.min():
        rth_scores = (r - r.min()) / (r.max() - r.min())
    else:
        rth_scores = np.ones_like(r)
    
    # Hitung skor final dengan bobot
    final = (
        WEIGHTS["price"] * price_scores +
        WEIGHTS["flood"] * flood_scores +
        WEIGHTS["crowd"] * crowd_scores +
        WEIGHTS["prox"] * prox_scores +
        WEIGHTS["rth"] * rth_scores
    )
    
    # Copy dataframe dan tambahkan skor
    df2 = df.copy()
    df2["score"] = final
    df2["price_score"] = price_scores
    df2["flood_score"] = flood_scores
    df2["crowd_score"] = crowd_scores
    df2["prox_score"] = prox_scores
    df2["rth_score"] = rth_scores
    
    logger.info(f"✅ Skor berhasil dihitung untuk {len(df2)} lokasi")
    return df2.sort_values("score", ascending=False)

# ============ IMAGE HANDLING ============

def sanitize_filename(name: str) -> str:
    """
    Membersihkan nama file untuk pencarian gambar.
    
    Args:
        name: Nama lokasi
        
    Returns:
        Nama file yang sudah dibersihkan
    """
    s = name.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s + ".jpg"

def possible_image_paths(name: str) -> List[str]:
    """
    Generate daftar kemungkinan path gambar untuk lokasi.
    
    Args:
        name: Nama lokasi
        
    Returns:
        List path yang mungkin
    """
    base_names = [
        sanitize_filename(name),
        name.lower().replace(" ", "") + ".jpg",
        name.lower().replace(" ", "-") + ".jpg",
        name + ".jpg"
    ]
    
    paths = []
    for base in base_names:
        # Root directory
        paths.append(os.path.join(BASE_DIR, base))
        # Static images directory
        paths.append(os.path.join(BASE_DIR, "static", "images", "lokasi", base))
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(paths))

def find_existing_image(name: str) -> Optional[str]:
    """
    Mencari gambar yang ada untuk lokasi tertentu.
    
    Args:
        name: Nama lokasi
        
    Returns:
        Path gambar jika ditemukan, None jika tidak
    """
    try:
        for p in possible_image_paths(name):
            if os.path.exists(p):
                logger.info(f"✅ Gambar ditemukan: {p}")
                return p
        logger.warning(f"⚠️ Gambar tidak ditemukan untuk: {name}")
        return None
    except Exception as e:
        logger.error(f"Error saat mencari gambar {name}: {e}")
        return None

# ============ FORMATTING FUNCTIONS ============

def format_total_price(price: float) -> str:
    """
    Format harga dalam juta atau miliar.
    
    Args:
        price: Harga dalam rupiah
        
    Returns:
        String harga yang terformat
    """
    juta = price / 1_000_000
    if juta < 1000:
        return f"{juta:.0f} juta"
    return f"{juta/1000:.1f} miliar"

# ============ ANALYSIS FUNCTIONS ============

def analyze_advantages_disadvantages(row: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Menganalisis kelebihan dan kekurangan suatu lokasi.
    
    Args:
        row: Series data lokasi
        
    Returns:
        Tuple (kelebihan, kekurangan)
    """
    advantages = []
    disadvantages = []
    
    # PRIORITAS 1: Analisis berdasarkan data DINAMIS (dari perhitungan)
    
    # Analisis HARGA (paling penting, bobot 40%)
    price_score = row.get("price_score", 0)
    if price_score > 0.7:
        advantages.append("Harga sangat terjangkau dibanding lokasi lain.")
    elif price_score > PRICE_SCORE_THRESHOLD:
        advantages.append("Harga relatif murah dibanding kecamatan lain.")
    elif price_score < 0.3:
        disadvantages.append("Harga cenderung mahal.")
    
    # Analisis BANJIR (bobot 30%)
    if row["flood_risk"] == "low":
        advantages.append("Area ini memiliki risiko banjir yang rendah.")
    elif row["flood_risk"] == "high":
        disadvantages.append("Berpotensi terdampak banjir.")
    
    # Analisis KERAMAIAN (bobot 15%)
    if row["crowd_level"] == "low":
        advantages.append("Lingkungan sekitar tenang.")
    elif row["crowd_level"] == "high":
        disadvantages.append("Keramaian area sekitar tinggi — kurang nyaman.")
    
    # Analisis AKSES PUBLIK (bobot 10%)
    if row["proximity_public"] == "high":
        advantages.append("Dekat dengan fasilitas umum.")
    elif row["proximity_public"] == "low":
        disadvantages.append("Akses fasilitas umum terbatas.")
    
    # Analisis RTH (bobot 5%)
    if row["rth_percent"] >= RTH_THRESHOLD_HIGH:
        advantages.append("RTH luas dan memadai.")
    elif row["rth_percent"] < RTH_THRESHOLD_LOW:
        disadvantages.append("RTH rendah — potensi area padat.")
    
    # PRIORITAS 2: Tambahkan info dari database
    key = row["name"].strip().lower().replace(" ", "").replace("-", "")
    if key in LOCATION_INFO:
        info = LOCATION_INFO[key]
        
        # Filter kelebihan: jangan tambah jika sudah ada info harga dari analisis dinamis
        for adv in info.get("kelebihan", []):
            # Skip jika berbicara tentang harga (sudah dihandle di atas)
            if "harga" in adv.lower() or "murah" in adv.lower() or "mahal" in adv.lower():
                continue
            # Skip jika duplikat
            if adv not in advantages:
                advantages.append(adv)
        
        # Filter kekurangan: jangan tambah jika kontradiksi dengan analisis dinamis
        for dis in info.get("kekurangan", []):
            # Skip jika berbicara tentang harga (sudah dihandle di atas)
            if "harga" in dis.lower() or "murah" in dis.lower() or "mahal" in dis.lower():
                continue
            # Skip jika duplikat
            if dis not in disadvantages:
                disadvantages.append(dis)
    
    return advantages, disadvantages

# ============ STREAMLIT UI ============

def main():
    """Fungsi utama aplikasi Streamlit"""
    
    st.set_page_config(layout="wide", page_title="Rekomendasi Tanah Bandung")
    
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
    
    # Input dari user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget_miliar = st.number_input(
            "💰 Masukkan Budget (dalam MILIAR)",
            min_value=MIN_BUDGET_MILIAR,
            max_value=MAX_BUDGET_MILIAR,
            step=0.1,
            value=1.0,
            help="Masukkan budget Anda dalam satuan miliar rupiah"
        )
        budget = budget_miliar * 1_000_000_000
    
    with col2:
        luas = st.number_input(
            "📏 Masukkan Luas Tanah (m²)",
            min_value=MIN_LUAS,
            step=5,
            value=100,
            help="Masukkan luas tanah yang diinginkan dalam meter persegi"
        )
    
    # Load data
    df = read_locations()
    max_loc = len(df)
    
    with col3:
        jumlah_rekom = st.number_input(
            "🔢 Jumlah lokasi yang ingin dianalisis:",
            min_value=1,
            max_value=max_loc,
            value=min(10, max_loc),
            step=1,
            help=f"Pilih 1-{max_loc} lokasi untuk dianalisis"
        )
    
    # Tampilkan bobot
    st.markdown("### 📘 Bobot Penilaian Lokasi")
    st.markdown(f"""
    - **Harga tanah:** {int(WEIGHTS['price']*100)}% (semakin murah → skor lebih tinggi)  
    - **Risiko banjir:** {int(WEIGHTS['flood']*100)}% (rendah → skor lebih tinggi)  
    - **Tingkat keramaian:** {int(WEIGHTS['crowd']*100)}% (low → skor lebih tinggi)  
    - **Akses fasilitas publik:** {int(WEIGHTS['prox']*100)}% (high → skor lebih tinggi)  
    - **RTH:** {int(WEIGHTS['rth']*100)}% (persentase RTH lebih besar → skor lebih tinggi)  
    """)
    st.caption("Penjelasan: skor akhir dihitung dengan menggabungkan kelima kriteria di atas sesuai bobot. Grafik menampilkan skor akhir dalam persen (0–100%).")
    
    # Tombol analisis
    if st.button("🔍 Tampilkan Rekomendasi", type="primary"):
        with st.spinner("Menganalisis lokasi..."):
            # Compute scores
            scored = compute_scores(df)
            scored["total_price"] = scored["price_per_m2"] * luas
            
            # Filter berdasarkan budget
            affordable = scored[scored["total_price"] <= budget]
            
            if affordable.empty:
                st.error(f"""
                ❌ Tidak ada lokasi yang sesuai dengan budget Anda!
                
                **Budget Anda:** {format_total_price(budget)}  
                **Luas:** {luas} m²
                
                💡 Saran:
                - Tingkatkan budget, atau
                - Kurangi luas tanah yang diinginkan
                """)
                st.stop()
            
            # Ambil top K
            topk = affordable.head(jumlah_rekom).reset_index(drop=True)
            top3 = topk.head(3).reset_index(drop=True)
            
            logger.info(f"✅ Ditemukan {len(affordable)} lokasi terjangkau")
            
            # Prepare display table
            display_tbl = topk[[
                "name", "price_per_m2_million", "total_price",
                "flood_risk", "crowd_level", "proximity_public",
                "rth_percent"
            ]].copy()
            
            display_tbl = display_tbl.rename(columns={
                "name": "Nama",
                "price_per_m2_million": "Harga_per_m2",
                "total_price": "Harga_total",
                "flood_risk": "Risiko_Banjir",
                "crowd_level": "Tingkat_Keramaian",
                "proximity_public": "Lokasi_Strategis",
                "rth_percent": "RTH (%)",
            })
            
            display_tbl["Harga_total"] = display_tbl["Harga_total"].apply(format_total_price)
            display_tbl["Harga_per_m2"] = display_tbl["Harga_per_m2"].apply(lambda x: f"{x:.0f} juta/m²")
            display_tbl["RTH (%)"] = display_tbl["RTH (%)"].apply(lambda x: f"{x:.0f}%")
            
            # Display table
            st.subheader(f"📌 {len(topk)} Lokasi yang Dianalisis (sesuai budget)")
            st.dataframe(display_tbl, use_container_width=True)
            
            # Display top 3
            st.subheader("🏆 3 Rekomendasi Lokasi Terbaik")
            
            for i, row in top3.iterrows():
                st.markdown(f"### 📍 {row['name']}")
                
                # Tampilkan gambar
                img_path = find_existing_image(row["name"])
                if img_path:
                    st.image(img_path, width=400)
                else:
                    if row["name"].strip().lower() == "cidadap":
                        st.markdown("""
                        **📘 Deskripsi Lokasi Cidadap (Foto tidak tersedia)**  
                        - 60% wilayah berupa dataran datar hingga berombak  
                        - Ketinggian sekitar 750 mdpl  
                        - Suhu harian 19°C – 28°C    
                        """)
                    else:
                        st.info("📷 Foto lokasi belum tersedia.")
                
                # Detail informasi
                st.write(f"- **Harga per m²:** {row['price_per_m2_million']:.0f} juta/m²")
                st.write(f"- **Harga total:** {format_total_price(row['total_price'])}")
                st.write(f"- **Risiko banjir:** {row['flood_risk']}")
                st.write(f"- **Tingkat keramaian:** {row['crowd_level']}")
                st.write(f"- **Akses fasilitas publik:** {row['proximity_public']}")
                st.write(f"- **RTH:** {row['rth_percent']:.0f}%")
                
                # Analisis kelebihan & kekurangan
                advantages, disadvantages = analyze_advantages_disadvantages(row)
                
                st.markdown("#### 🟢 Kelebihan:")
                if advantages:
                    for a in advantages:
                        st.markdown(f"- {a}")
                else:
                    st.markdown("- (Tidak ada catatan kelebihan spesifik.)")
                
                st.markdown("#### 🔴 Kekurangan:")
                if disadvantages:
                    for d in disadvantages:
                        st.markdown(f"- {d}")
                else:
                    st.markdown("- (Tidak ada catatan kekurangan spesifik.)")
                
                st.markdown("---")
            
            # Bar Chart
            st.subheader("📊 Perbandingan Antar Kecamatan (Top 3)")
            
            labels = top3["name"].tolist()
            scores_pct = (top3["score"].values * 100).round(1)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(labels, scores_pct, width=0.5, color=['#4CAF50', '#2196F3', '#FF9800'])
            
            ax.set_ylim(0, 100)
            ax.set_ylabel("Skor (%)", fontsize=12)
            ax.set_title("Skor Total Lokasi (dalam %) — Semakin tinggi semakin direkomendasikan", fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, h in zip(bars, scores_pct):
                ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            **Keterangan skor:** Skor akhir dihitung dari kombinasi kriteria berikut dengan bobot:
            - **Harga tanah:** {int(WEIGHTS['price']*100)}%
            - **Risiko banjir:** {int(WEIGHTS['flood']*100)}%
            - **Tingkat keramaian:** {int(WEIGHTS['crowd']*100)}%
            - **Akses fasilitas publik:** {int(WEIGHTS['prox']*100)}%
            - **RTH:** {int(WEIGHTS['rth']*100)}%
            """)
            st.caption("Contoh interpretasi: Nilai 78% artinya lokasi memperoleh skor total 0.78 berdasarkan bobot di atas.")
            
            # Radar Chart
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
            angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            fig = plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            
            colors = ['#4CAF50', '#2196F3', '#FF9800']
            for i, loc in enumerate(top3["name"]):
                v = values[i]
                v = v + v[:1]
                ax.plot(angles, v, linewidth=2, label=loc, color=colors[i])
                ax.fill(angles, v, alpha=0.15, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_yticks([])
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            
            plt.title("Perbandingan Kriteria Lokasi (Radar Chart)", size=14, pad=20, fontweight='bold')
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
            
            st.pyplot(fig)
            
            st.success("✅ Analisis selesai! Scroll ke atas untuk melihat hasil lengkap.")

if __name__ == "__main__":
    main()


