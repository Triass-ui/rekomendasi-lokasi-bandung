import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import logging
from typing import Dict, List, Optional, Tuple

# Pengaturan logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ KONSTANTA ============
DIREKTORI_DASAR = os.path.dirname(os.path.abspath(__file__))

# Bobot penilaian
BOBOT = {
    "harga": 0.40,
    "banjir": 0.30,
    "keramaian": 0.15,
    "akses": 0.10,
    "rth": 0.05


# Konstanta pemetaan kategori
PETA_BANJIR = {"low": 1.0, "medium": 0.5, "high": 0.0}
PETA_KERAMAIAN = {"low": 0.0, "medium": 0.5, "high": 1.0}
PETA_AKSES = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Ambang batas RTH
AMBANG_RTH_TINGGI = 25
AMBANG_RTH_RENDAH = 15

# Ambang batas skor harga untuk kelebihan
AMBANG_SKOR_HARGA = 0.6

# Batas budget
BUDGET_MINIMAL_MILIAR = 0.1
BUDGET_MAKSIMAL_MILIAR = 1000.0
LUAS_MINIMAL = 20

IND_KE_EN = {
    "rendah": "low",
    "sedang": "medium",
    "tinggi": "high",
    "low": "low",
    "medium": "medium",
    "high": "high"
}

# ============ INFORMASI LOKASI ============
INFO_LOKASI: Dict[str, Dict[str, List[str]]] = {
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

# ============ MEMUAT DATA ============

@st.cache_data
def baca_lokasi(jalur_xlsx: str = "locations.xlsx", jalur_csv: str = "locations.csv") -> pd.DataFrame:
    """
    Membaca dan memproses data lokasi dari file Excel atau CSV.
    
    Args:
        jalur_xlsx: Jalur ke file Excel
        jalur_csv: Jalur ke file CSV
        
    Returns:
        DataFrame yang sudah diproses
    """
    try:
        # Coba baca Excel dulu
        jalur_xlsx_lengkap = os.path.join(DIREKTORI_DASAR, jalur_xlsx)
        jalur_csv_lengkap = os.path.join(DIREKTORI_DASAR, jalur_csv)
        
        if os.path.exists(jalur_xlsx_lengkap):
            df = pd.read_excel(jalur_xlsx_lengkap)
            logger.info(f"✅ Berhasil membaca file: {jalur_xlsx}")
        elif os.path.exists(jalur_csv_lengkap):
            df = pd.read_csv(jalur_csv_lengkap)
            logger.info(f"✅ Berhasil membaca file: {jalur_csv}")
        else:
            pesan_error = f"""
            ❌ File tidak ditemukan!
            
            Pastikan salah satu file berikut ada di folder:
            📁 {DIREKTORI_DASAR}
            
            File yang dicari:
            - {jalur_xlsx} ATAU
            - {jalur_csv}
            
            Silakan upload file terlebih dahulu.
            """
            st.error(pesan_error)
            logger.error(f"File tidak ditemukan di {DIREKTORI_DASAR}")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {str(e)}")
        logger.error(f"Error membaca file: {e}")
        st.stop()
    
    # Normalisasi nama kolom
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Pemetaan nama kolom
    peta_kolom = {
        "nama": "name",
        "harga_per_m2": "price_per_m2",
        "resiko_banjir": "flood_risk",
        "tingkat_keramaian": "crowd_level",
        "persentase_rth": "rth_percent",
        "lokasi_strategis": "proximity_public"
    }
    df = df.rename(columns=peta_kolom)
    
    # Validasi kolom yang diperlukan
    kolom_wajib = ["name", "price_per_m2", "flood_risk", "crowd_level", "rth_percent", "proximity_public"]
    kolom_hilang = [k for k in kolom_wajib if k not in df.columns]
    if kolom_hilang:
        st.error(f"❌ Kolom berikut hilang di file: {', '.join(kolom_hilang)}")
        logger.error(f"Kolom hilang: {kolom_hilang}")
        st.stop()
    
    # Petakan kategori
    def petakan_kategori(nilai) -> str:
        """Petakan kategori dengan default aman"""
        if pd.isna(nilai):
            return "medium"
        v = str(nilai).strip().lower()
        return IND_KE_EN.get(v, "medium")
    
    df["flood_risk"] = df["flood_risk"].apply(petakan_kategori)
    df["crowd_level"] = df["crowd_level"].apply(petakan_kategori)
    df["proximity_public"] = df["proximity_public"].apply(petakan_kategori)
    
    # Konversi harga dan RTH
    df["price_per_m2_million"] = pd.to_numeric(df["price_per_m2"], errors="coerce").fillna(0)
    df["price_per_m2"] = df["price_per_m2_million"] * 1_000_000
    df["rth_percent"] = pd.to_numeric(df["rth_percent"], errors="coerce").fillna(0)
    
    logger.info(f"✅ Berhasil memproses {len(df)} lokasi")
    return df

# ============ FUNGSI PENILAIAN ============

def normalisasi_skor_harga(df: pd.DataFrame) -> np.ndarray:
    """
    Normalisasi skor harga: semakin murah semakin tinggi skornya.
    
    Args:
        df: DataFrame dengan kolom price_per_m2
        
    Returns:
        Array skor yang dinormalisasi (0-1)
    """
    harga = df["price_per_m2"].values.astype(float)
    minimal, maksimal = harga.min(), harga.max()
    
    if minimal == maksimal:
        logger.warning("Semua harga sama, mengembalikan skor seragam")
        return np.ones_like(harga)
    
    return (maksimal - harga) / (maksimal - minimal)

@st.cache_data
def hitung_skor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung skor total untuk setiap lokasi berdasarkan kriteria.
    
    Args:
        df: DataFrame dengan data lokasi
        
    Returns:
        DataFrame dengan kolom skor tambahan
    """
    # Hitung skor individual
    skor_harga = normalisasi_skor_harga(df)
    skor_banjir = df["flood_risk"].map(PETA_BANJIR).fillna(0.5).values
    skor_keramaian = df["crowd_level"].map(PETA_KERAMAIAN).fillna(0.5).values
    skor_akses = df["proximity_public"].map(PETA_AKSES).fillna(0.5).values
    
    # Normalisasi RTH
    r = df["rth_percent"].values.astype(float)
    if r.max() != r.min():
        skor_rth = (r - r.min()) / (r.max() - r.min())
    else:
        skor_rth = np.ones_like(r)
    
    # Hitung skor final dengan bobot
    skor_final = (
        BOBOT["harga"] * skor_harga +
        BOBOT["banjir"] * skor_banjir +
        BOBOT["keramaian"] * skor_keramaian +
        BOBOT["akses"] * skor_akses +
        BOBOT["rth"] * skor_rth
    )
    df2 = df.copy()
    df2["score"] = skor_final
    df2["price_score"] = skor_harga
    df2["flood_score"] = skor_banjir
    df2["crowd_score"] = skor_keramaian
    df2["prox_score"] = skor_akses
    df2["rth_score"] = skor_rth
    
    logger.info(f"✅ Skor berhasil dihitung untuk {len(df2)} lokasi")
    return df2.sort_values("score", ascending=False)

# ============ PENANGANAN GAMBAR ============

def bersihkan_nama_file(nama: str) -> str:
    """
    Membersihkan nama file untuk pencarian gambar.
    
    Args:
        nama: Nama lokasi
        
    Returns:
        Nama file yang sudah dibersihkan
    """
    s = nama.lower().strip()
    s = s.replace(" ", "_")
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s + ".jpg"

def kemungkinan_jalur_gambar(nama: str) -> List[str]:
    """
    Generate daftar kemungkinan jalur gambar untuk lokasi.
    
    Args:
        nama: Nama lokasi
        
    Returns:
        List jalur yang mungkin
    """
    nama_dasar = [
        bersihkan_nama_file(nama),
        nama.lower().replace(" ", "") + ".jpg",
        nama.lower().replace(" ", "-") + ".jpg",
        nama + ".jpg"
    ]
    
    jalur = []
    for nd in nama_dasar:
        jalur.append(os.path.join(DIREKTORI_DASAR, nd))
        jalur.append(os.path.join(DIREKTORI_DASAR, "static", "images", "lokasi", nd))
    return list(dict.fromkeys(jalur))

def cari_gambar_tersedia(nama: str) -> Optional[str]:
    """
    Mencari gambar yang ada untuk lokasi tertentu.
    
    Args:
        nama: Nama lokasi
        
    Returns:
        Jalur gambar jika ditemukan, None jika tidak
    """
    try:
        for j in kemungkinan_jalur_gambar(nama):
            if os.path.exists(j):
                logger.info(f"✅ Gambar ditemukan: {j}")
                return j
        logger.warning(f"⚠️ Gambar tidak ditemukan untuk: {nama}")
        return None
    except Exception as e:
        logger.error(f"Error saat mencari gambar {nama}: {e}")
        return None

# ============ FUNGSI PEMFORMATAN ============

def format_harga_total(harga: float) -> str:
    """
    Format harga dalam juta atau miliar.
    
    Args:
        harga: Harga dalam rupiah
        
    Returns:
        String harga yang terformat
    """
    juta = harga / 1_000_000
    if juta < 1000:
        return f"{juta:.0f} juta"
    return f"{juta/1000:.1f} miliar"

# ============ FUNGSI ANALISIS ============

def analisis_kelebihan_kekurangan(baris: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Menganalisis kelebihan dan kekurangan suatu lokasi.
    
    Args:
        baris: Series data lokasi
        
    Returns:
        Tuple (kelebihan, kekurangan)
    """
    kelebihan = []
    kekurangan = []
    
    # PRIORITAS 1: Analisis berdasarkan data DINAMIS (dari perhitungan)
    
    # Analisis HARGA (paling penting, bobot 40%)
    skor_harga = baris.get("price_score", 0)
    if skor_harga > 0.7:
        kelebihan.append("Harga sangat terjangkau dibanding lokasi lain.")
    elif skor_harga > AMBANG_SKOR_HARGA:
        kelebihan.append("Harga relatif murah dibanding kecamatan lain.")
    elif skor_harga < 0.3:
        kekurangan.append("Harga cenderung mahal.")
    
    # Analisis BANJIR (bobot 30%)
    if baris["flood_risk"] == "low":
        kelebihan.append("Area ini memiliki risiko banjir yang rendah.")
    elif baris["flood_risk"] == "high":
        kekurangan.append("Berpotensi terdampak banjir.")
    
    # Analisis KERAMAIAN (bobot 15%)
    if baris["crowd_level"] == "low":
        kelebihan.append("Lingkungan sekitar tenang.")
    elif baris["crowd_level"] == "high":
        kekurangan.append("Keramaian area sekitar tinggi — kurang nyaman.")
    
    # Analisis AKSES PUBLIK (bobot 10%)
    if baris["proximity_public"] == "high":
        kelebihan.append("Dekat dengan fasilitas umum.")
    elif baris["proximity_public"] == "low":
        kekurangan.append("Akses fasilitas umum terbatas.")
    
    # Analisis RTH (bobot 5%)
    if baris["rth_percent"] >= AMBANG_RTH_TINGGI:
        kelebihan.append("RTH luas dan memadai.")
    elif baris["rth_percent"] < AMBANG_RTH_RENDAH:
        kekurangan.append("RTH rendah — potensi area padat.")
    
    # PRIORITAS 2: Tambahkan info dari database
    kunci = baris["name"].strip().lower().replace(" ", "").replace("-", "")
    if kunci in INFO_LOKASI:
        info = INFO_LOKASI[kunci]
        
        # Filter kelebihan: jangan tambah jika sudah ada info harga dari analisis dinamis
        for klb in info.get("kelebihan", []):
            if "harga" in klb.lower() or "murah" in klb.lower() or "mahal" in klb.lower():
                continue
            if klb not in kelebihan:
                kelebihan.append(klb)
        
        # Filter kekurangan: jangan tambah jika kontradiksi dengan analisis dinamis
        for krg in info.get("kekurangan", []):
            if "harga" in krg.lower() or "murah" in krg.lower() or "mahal" in krg.lower():
                continue
            if krg not in kekurangan:
                kekurangan.append(krg)
    
    return kelebihan, kekurangan

# ============ ANTARMUKA STREAMLIT ============

def utama():
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
    
    # Input dari pengguna
    kol1, kol2, kol3 = st.columns(3)
    
    with kol1:
        budget_miliar = st.number_input(
            "💰 Masukkan Budget (dalam MILIAR)",
            min_value=BUDGET_MINIMAL_MILIAR,
            max_value=BUDGET_MAKSIMAL_MILIAR,
            step=0.1,
            value=1.0,
            help="Masukkan budget Anda dalam satuan miliar rupiah"
        )
        budget = budget_miliar * 1_000_000_000
    
    with kol2:
        luas = st.number_input(
            "📏 Masukkan Luas Tanah (m²)",
            min_value=LUAS_MINIMAL,
            step=5,
            value=100,
            help="Masukkan luas tanah yang diinginkan dalam meter persegi"
        )
    
    # Muat data
    df = baca_lokasi()
    lokasi_maks = len(df)
    
    with kol3:
        jumlah_rekom = st.number_input(
            "🔢 Jumlah lokasi yang ingin dianalisis:",
            min_value=1,
            max_value=lokasi_maks,
            value=min(10, lokasi_maks),
            step=1,
            help=f"Pilih 1-{lokasi_maks} lokasi untuk dianalisis"
        )
    
    # Tampilkan bobot
    st.markdown("### 📘 Bobot Penilaian Lokasi")
    st.markdown(f"""
    - **Harga tanah:** {int(BOBOT['harga']*100)}% (semakin murah → skor lebih tinggi)  
    - **Risiko banjir:** {int(BOBOT['banjir']*100)}% (rendah → skor lebih tinggi)  
    - **Tingkat keramaian:** {int(BOBOT['keramaian']*100)}% (rendah → skor lebih tinggi)  
    - **Akses fasilitas publik:** {int(BOBOT['akses']*100)}% (tinggi → skor lebih tinggi)  
    - **RTH:** {int(BOBOT['rth']*100)}% (persentase RTH lebih besar → skor lebih tinggi)  
    """)
    st.caption("Penjelasan: skor akhir dihitung dengan menggabungkan kelima kriteria di atas sesuai bobot. Grafik menampilkan skor akhir dalam persen (0–100%).")
    
    # Tombol analisis
    if st.button("🔍 Tampilkan Rekomendasi", type="primary"):
        with st.spinner("Menganalisis lokasi..."):
            # Hitung skor
            hasil_skor = hitung_skor(df)
            hasil_skor["total_price"] = hasil_skor["price_per_m2"] * luas
            
            # Filter berdasarkan budget
            terjangkau = hasil_skor[hasil_skor["total_price"] <= budget]
            
            if terjangkau.empty:
                st.error(f"""
                ❌ Tidak ada lokasi yang sesuai dengan budget Anda!
                
                **Budget Anda:** {format_harga_total(budget)}  
                **Luas:** {luas} m²
                
                💡 Saran:
                - Tingkatkan budget, atau
                - Kurangi luas tanah yang diinginkan
                """)
                st.stop()
            
            # Ambil top K
            topk = terjangkau.head(jumlah_rekom).reset_index(drop=True)
            top3 = topk.head(3).reset_index(drop=True)
            
            logger.info(f"✅ Ditemukan {len(terjangkau)} lokasi terjangkau")
            
            # Tampilkan tabel
            tabel_tampil = topk[[
                "name", "price_per_m2_million", "total_price",
                "flood_risk", "crowd_level", "proximity_public",
                "rth_percent"
            ]].copy()
            
            tabel_tampil = tabel_tampil.rename(columns={
                "name": "Nama",
                "price_per_m2_million": "Harga_per_m2",
                "total_price": "Harga_total",
                "flood_risk": "Risiko_Banjir",
                "crowd_level": "Tingkat_Keramaian",
                "proximity_public": "Lokasi_Strategis",
                "rth_percent": "RTH (%)",
            })
            
            tabel_tampil["Harga_total"] = tabel_tampil["Harga_total"].apply(format_harga_total)
            tabel_tampil["Harga_per_m2"] = tabel_tampil["Harga_per_m2"].apply(lambda x: f"{x:.0f} juta/m²")
            tabel_tampil["RTH (%)"] = tabel_tampil["RTH (%)"].apply(lambda x: f"{x:.0f}%")
            
            # Tampilkan tabel
            st.subheader(f"📌 {len(topk)} Lokasi yang Dianalisis (sesuai budget)")
            st.dataframe(tabel_tampil, use_container_width=True)
            
            # Tampilkan top 3
            st.subheader("🏆 3 Rekomendasi Lokasi Terbaik")
            
            for i, baris in top3.iterrows():
                st.markdown(f"### 📍 {baris['name']}")
                
                # Tampilkan gambar
                jalur_gambar = cari_gambar_tersedia(baris["name"])
                if jalur_gambar:
                    st.image(jalur_gambar, width=400)
                else:
                    if baris["name"].strip().lower() == "cidadap":
                        st.markdown("""
                        **📘 Deskripsi Lokasi Cidadap (Foto tidak tersedia)**  
                        - 60% wilayah berupa dataran datar hingga berombak  
                        - Ketinggian sekitar 750 mdpl  
                        - Suhu harian 19°C – 28°C    
                        """)
                    else:
                        st.info("📷 Foto lokasi belum tersedia.")
                
                # Detail informasi
                st.write(f"- **Harga per m²:** {baris['price_per_m2_million']:.0f} juta/m²")
                st.write(f"- **Harga total:** {format_harga_total(baris['total_price'])}")
                st.write(f"- **Risiko banjir:** {baris['flood_risk']}")
                st.write(f"- **Tingkat keramaian:** {baris['crowd_level']}")
                st.write(f"- **Akses fasilitas publik:** {baris['proximity_public']}")
                st.write(f"- **RTH:** {baris['rth_percent']:.0f}%")
                
                # Analisis kelebihan & kekurangan
                kelebihan, kekurangan = analisis_kelebihan_kekurangan(baris)
                
                st.markdown("#### 🟢 Kelebihan:")
                if kelebihan:
                    for k in kelebihan:
                        st.markdown(f"- {k}")
                else:
                    st.markdown("- (Tidak ada catatan kelebihan spesifik.)")
                
                st.markdown("#### 🔴 Kekurangan:")
                if kekurangan:
                    for k in kekurangan:
                        st.markdown(f"- {k}")
                else:
                    st.markdown("- (Tidak ada catatan kekurangan spesifik.)")
                
                st.markdown("---")
            
            # Grafik Batang
            st.subheader("📊 Perbandingan Antar Kecamatan (Top 3)")
            
            label = top3["name"].tolist()
            skor_persen = (top3["score"].values * 100).round(1)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            batang = ax.bar(label, skor_persen, width=0.5, color=['#4CAF50', '#2196F3', '#FF9800'])
            
            ax.set_ylim(0, 100)
            ax.set_ylabel("Skor (%)", fontsize=12)
            ax.set_title("Skor Total Lokasi (dalam %) — Semakin tinggi semakin direkomendasikan", fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, h in zip(batang, skor_persen):
                ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            **Keterangan skor:** Skor akhir dihitung dari kombinasi kriteria berikut dengan bobot:
            - **Harga tanah:** {int(BOBOT['harga']*100)}%
            - **Risiko banjir:** {int(BOBOT['banjir']*100)}%
            - **Tingkat keramaian:** {int(BOBOT['keramaian']*100)}%
            - **Akses fasilitas publik:** {int(BOBOT['akses']*100)}%
            - **RTH:** {int(BOBOT['rth']*100)}%
            """)
            st.caption("Contoh interpretasi: Nilai 78% artinya lokasi memperoleh skor total 0.78 berdasarkan bobot di atas.")
            
            # Grafik Radar
            st.subheader("Grafik Radar Perbandingan Kriteria (Top 3)")
            
            kategori = ["Harga Lahan", "Risiko Banjir", "Tingkat Keramaian", "Akses Publik", "RTH (%)"]
            
            nilai = []
            for _, baris in top3.iterrows():
                skor_risiko_banjir = 1 - baris["flood_score"]
                nilai.append([
                    baris["price_score"],
                    skor_risiko_banjir,
                    baris["crowd_score"],
                    baris["prox_score"],
                    baris["rth_score"]
                ])
            
            jumlah_var = len(kategori)
            sudut = np.linspace(0, 2*np.pi, jumlah_var, endpoint=False).tolist()
            sudut += sudut[:1]
            
            fig = plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            
            warna = ['#4CAF50', '#2196F3', '#FF9800']
            for i, lok in enumerate(top3["name"]):
                v = nilai[i]
                v = v + v[:1]
                ax.plot(sudut, v, linewidth=2, label=lok, color=warna[i])
                ax.fill(sudut, v, alpha=0.15, color=warna[i])
            
            ax.set_xticks(sudut[:-1])
            ax.set_xticklabels(kategori, size=10)
            ax.set_yticks([])
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            
            plt.title("Perbandingan Kriteria Lokasi (Grafik Radar)", size=14, pad=20, fontweight='bold')
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
            
            st.pyplot(fig)
            
            st.success("✅ Analisis selesai! Apakah anda sudah menentukan hasilnya?")

if __name__ == "__main__":
    utama()

