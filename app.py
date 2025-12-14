import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------------
# 1. AYARLAR VE MODEL YÜKLEME
# --------------------------------------------------------
st.set_page_config(
    page_title="İstanbul Emlak & Yatırım",
    page_icon="🏢",
    layout="wide"
)


@st.cache_resource
def model_yukle():
    # Dosya yolları listesi
    yollar = [
        'models/house_price_model.pkl',
        'house_price_model.pkl'
    ]

    for yol in yollar:
        if os.path.exists(yol):
            return joblib.load(yol)

    return None


model = model_yukle()

# EMNİYET KİLİDİ
if model is None:
    st.error("🚨 HATA: Model dosyası bulunamadı!")
    st.warning(
        "Lütfen 'app.py' dosyasının yanında 'model' klasörü "
        "ve içinde '.pkl' dosyası olduğundan emin olun."
    )
    st.stop()

# İlçe Listesi
ILCELER = [
    'Adalar', 'Arnavutköy', 'Ataşehir', 'Avcılar', 'Bağcılar', 'Bahçelievler',
    'Bakırköy', 'Başakşehir', 'Bayrampaşa', 'Beşiktaş', 'Beykoz', 'Beylikdüzü',
    'Beyoğlu', 'Büyükçekmece', 'Çatalca', 'Çekmeköy', 'Esenler', 'Esenyurt',
    'Eyüpsultan', 'Fatih', 'Gaziosmanpaşa', 'Güngören', 'Kadıköy', 'Kağıthane',
    'Kartal', 'Küçükçekmece', 'Maltepe', 'Pendik', 'Sancaktepe', 'Sarıyer',
    'Silivri', 'Sultanbeyli', 'Sultangazi', 'Şile', 'Şişli', 'Tuzla',
    'Ümraniye', 'Üsküdar', 'Zeytinburnu'
]

# --------------------------------------------------------
# 2. SOL MENÜ (GİRDİLER)
# --------------------------------------------------------
st.sidebar.title("🏢 Emlak Parametreleri")
secilen_ilce = st.sidebar.selectbox("İlçe", ILCELER)
m2_gross = st.sidebar.number_input("Brüt m²", 40, 1000, 100)
m2_net = st.sidebar.number_input("Net m²", 30, 900, 85)
oda_sayisi = st.sidebar.slider("Oda", 1, 10, 3)
bina_yasi = st.sidebar.slider("Yaş", 0, 50, 5)
kat_sayisi = st.sidebar.number_input("Bina Katı", 1, 50, 5)
bulundugu_kat = st.sidebar.number_input("Daire Katı", 0, 50, 2)
banyo = st.sidebar.radio("Banyo", [1, 2, 3, 4], horizontal=True)

with st.sidebar.expander("Detay Özellikler"):
    krediye_uygun = st.checkbox("Kredi Uygun", True)
    otopark = st.checkbox("Otopark", True)
    asansor = st.checkbox("Asansör", True)
    balkon = st.checkbox("Balkon", True)
    esyali = st.checkbox("Eşyalı", False)
    isitma_list = [
        'Kombi', 'Merkezi Sistem', 'Yerden Isıtma',
        'Klima/Elektrikli'
    ]
    isitma = st.selectbox("Isıtma", isitma_list)
    kullanim = st.selectbox(
        "Durum", ['Mülk Sahibi Oturuyor', 'Kiracılı', 'Boş'])

# --------------------------------------------------------
# 3. HESAPLAMA MOTORU
# --------------------------------------------------------
st.title("📈 Emlak Değerleme ve Yatırım Analizi")

# Sekmeler
tab1, tab2 = st.tabs(["🏠 Değerleme Analizi", "💰 Yatırımcı Paneli"])


def tahmin_et():
    girdi = {
        'm2_gross': m2_gross, 'm2_net': m2_net, 'oda_sayisi': oda_sayisi,
        'bina_yasi': bina_yasi, 'bulundugu_kat': bulundugu_kat,
        'kat_sayisi': kat_sayisi, 'banyo_sayisi': banyo,
        'balkon': int(balkon), 'asansor': int(asansor),
        'esyali_mi': int(esyali), 'krediye_uygun': int(krediye_uygun),
        'otopark': int(otopark), 'ilce': secilen_ilce,
        'kullanim_durumu': kullanim, 'isitma_tipi': isitma
    }
    df = pd.DataFrame([girdi])
    df = pd.get_dummies(df)

    # Sütun eşitleme
    if hasattr(model, 'feature_names_in_'):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        st.warning("Model sütun bilgisi okunamadı!")

    return model.predict(df)[0]


tahmin_fiyat = tahmin_et()
sapma = 205000  # Model MAE değeri

# --- TAB 1: STANDART DEĞERLEME ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Adil Piyasa Değeri")
        deger_str = f"{tahmin_fiyat:,.0f} TL"
        st.metric(label="Tahmini Fiyat", value=deger_str, delta=None)

        alt_sinir = tahmin_fiyat - sapma
        ust_sinir = tahmin_fiyat + sapma
        st.info(f"Güven Aralığı: {alt_sinir:,.0f} TL - {ust_sinir:,.0f} TL")

    with col2:
        ilan_fiyati = st.number_input(
            "Karşılaştırma için İlan Fiyatı (TL)", 0, step=50000
        )
        if ilan_fiyati > 0:
            fark = ilan_fiyati - tahmin_fiyat
            abs_fark = abs(fark)

            if ilan_fiyati < tahmin_fiyat - sapma:
                msg = f"🔥 BÜYÜK FIRSAT! Değerinin {abs_fark:,.0f} TL altına."
                st.success(msg)
            elif ilan_fiyati < tahmin_fiyat:
                msg = f"✅ FIRSAT. Piyasa değerinin {abs_fark:,.0f} TL altına."
                st.success(msg)
            elif ilan_fiyati > tahmin_fiyat + sapma:
                msg = f"⚠️ PAHALI! Bu eve {fark:,.0f} TL fazla isteniyor."
                st.error(msg)
            else:
                msg = "⚖️ Normal. Fiyat piyasa ortalamasında."
                st.warning(msg)

# --- TAB 2: YATIRIMCI ANALİZİ ---
with tab2:
    st.header("ROI ve Kira Getirisi Simülasyonu")

    col_inv1, col_inv2, col_inv3 = st.columns(3)

    with col_inv1:
        amortisman_yili = st.slider(
            "Bölge Amortisman Süresi (Yıl)", 12, 30, 20
        )

    # Hesaplamalar
    tahmini_kira = tahmin_fiyat / (amortisman_yili * 12)
    yillik_getiri_orani = (tahmini_kira * 12) / tahmin_fiyat * 100

    with col_inv2:
        st.metric("Tahmini Aylık Kira", f"{tahmini_kira:,.0f} TL")
    with col_inv3:
        st.metric("Yıllık Kira Getirisi", f"%{yillik_getiri_orani:.2f}")

    st.divider()

    # 10 Yıllık Projeksiyon
    st.subheader("📊 10 Yıllık Değer Artış Tahmini")
    enflasyon_tahmini = st.slider(
        "Yıllık Beklenen Değer Artışı (%)", 10, 100, 40
    )

    gelecek_yillar = list(range(1, 11))
    gelecek_degerler = [
        tahmin_fiyat * ((1 + enflasyon_tahmini / 100) ** yil)
        for yil in gelecek_yillar
    ]

    chart_data = pd.DataFrame({
        'Yıl': gelecek_yillar,
        'Tahmini Değer (TL)': gelecek_degerler
    })

    st.line_chart(chart_data, x='Yıl', y='Tahmini Değer (TL)')

    # Sonucu güvenli yazdırma
    bes_yil_sonra = gelecek_degerler[4]
    yorum_metni = (
        f"💡 **Yorum:** Yıllık %{enflasyon_tahmini} artış senaryosunda, "
        f"bu ev 5 yıl sonra yaklaşık **{bes_yil_sonra:,.0f} TL** "
        f"değerine ulaşabilir."
    )
    st.write(yorum_metni)
