import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TurkDetect AI - Gemini Powered", layout="wide")

# --- YARDIMCI FONKSİYONLAR ---

def extract_names_from_chunk(names_chunk, api_key):
    """
    Bir grup (örn: 50 adet) ismi Gemini'ye gönderir ve Türk olanları JSON olarak ister.
    """
    try:
        genai.configure(api_key=api_key)
        
        # En hızlı ve ucuz model: Flash
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )

        prompt = f"""
        Act as a strictly deterministic data classifier.
        Below is a list of full names (First Name + Last Name).
        Identify which of these people are likely of TURKISH origin based on their names.
        
        Rules:
        1. Consider Turkish characters (ğ, ş, ı, ö, ü, ç) even if written in ASCII (g, s, i, o, u, c).
        2. Look for Turkish linguistic patterns in first names and last names (suffixes like -oglu, -kaya, -demi, -er, -kan).
        3. Be strict. If unsure or if it's a common western name (e.g., 'Sarah Jones'), do not include it.
        4. "Can" is a Turkish name, but check the last name to confirm it's not English context.
        
        Input List:
        {json.dumps(names_chunk)}

        Output Format:
        Return a JSON object with a single key "turkish_names" containing the list of full names found.
        Example: {{"turkish_names": ["Ahmet Yilmaz", "Selin Demir"]}}
        """

        response = model.generate_content(prompt)
        
        # JSON yanıtını parse et
        result = json.loads(response.text)
        return result.get("turkish_names", [])

    except Exception as e:
        st.error(f"API Hatası: {e}")
        return []

# --- ARAYÜZ ---
st.title("🤖 TurkDetect AI | Gemini API")
st.markdown("""
Bu araç, **Gemini 1.5 Flash** modelini kullanarak yüklenen CSV dosyasındaki kişilerin Türk olup olmadığını analiz eder.
Herhangi bir isim sözlüğü kullanmaz, doğrudan Yapay Zeka'nın kültürel bilgisini kullanır.
""")

# Sidebar: API Key ve Ayarlar
with st.sidebar:
    st.header("🔑 Kimlik Doğrulama")
    api_key = st.text_input("Google Gemini API Key", type="password", help="aistudio.google.com adresinden alabilirsiniz.")
    
    st.markdown("---")
    st.header("⚙️ Performans Ayarları")
    batch_size = st.slider("Batch Boyutu (Tek seferde sorulacak isim)", 20, 100, 50, help="Yüksek sayı daha hızlıdır ama model hata yapabilir.")
    request_delay = st.slider("İstek Gecikmesi (Saniye)", 0.0, 2.0, 0.5, help="Rate Limit yememek için bekleme süresi.")

# Ana Ekran
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Dosya Yükle")
    uploaded_file = st.file_uploader("Apollo/LinkedIn CSV (Max 50k)", type=["csv"])

    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        
        # Kolon Tespiti
        cols = [c.lower() for c in df.columns]
        fname_col = next((c for c in df.columns if c.lower() in ['first name', 'firstname', 'ad', 'name']), None)
        lname_col = next((c for c in df.columns if c.lower() in ['last name', 'lastname', 'soyad', 'surname']), None)

        if fname_col and lname_col:
            st.success(f"✅ {len(df)} satır yüklendi. Kolonlar bulundu.")
            
            # Tam İsim Kolonu Oluştur (AI'ya bunu göndereceğiz)
            df['Full_Name_Temp'] = df[fname_col].astype(str) + " " + df[lname_col].astype(str)
            all_names = df['Full_Name_Temp'].tolist()
            
            if st.button("🚀 AI Analizini Başlat"):
                identified_turkish_names = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Batch Processing Döngüsü
                total_batches = (len(all_names) + batch_size - 1) // batch_size
                
                for i in range(0, len(all_names), batch_size):
                    batch = all_names[i : i + batch_size]
                    
                    # API Çağrısı
                    found_names = extract_names_from_chunk(batch, api_key)
                    identified_turkish_names.extend(found_names)
                    
                    # UI Güncelleme
                    current_batch = (i // batch_size) + 1
                    progress = min(current_batch / total_batches, 1.0)
                    progress_bar.progress(progress)
                    status_text.code(f"İşleniyor: {current_batch}/{total_batches} Paket | Bulunan Türk: {len(identified_turkish_names)}")
                    
                    # Rate Limit Koruması
                    time.sleep(request_delay)

                # --- SONUÇ FİLTRELEME ---
                # AI'dan dönen isimleri orijinal veride işaretle
                # Performans için Set'e çeviriyoruz
                turkish_set = set(identified_turkish_names)
                
                # Orijinal dataframe'i filtrele
                result_df = df[df['Full_Name_Temp'].isin(turkish_set)].copy()
                
                # Geçici kolonu sil
                result_df.drop(columns=['Full_Name_Temp'], inplace=True)
                
                st.session_state['results'] = result_df
                st.session_state['processed'] = True

        else:
            st.error("CSV'de İsim/Soyisim kolonu bulunamadı.")
    elif uploaded_file and not api_key:
        st.warning("Lütfen sol menüden API Key giriniz.")

# Sonuç Ekranı (Session State ile kalıcı)
if st.session_state.get('processed'):
    res = st.session_state['results']
    with col2:
        st.subheader("🎯 Analiz Sonuçları")
        st.info(f"Toplam {len(res)} Türk profili tespit edildi.")
        
        st.dataframe(res, height=600)
        
        # Excel İndirme
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            res.to_excel(writer, index=False, sheet_name='Turkish Leads')
            
        st.download_button(
            label="📥 Excel Olarak İndir",
            data=buffer.getvalue(),
            file_name="gemini_filtered_leads.xlsx",
            mime="application/vnd.ms-excel"
        )
