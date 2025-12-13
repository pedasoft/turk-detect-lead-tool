import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import time
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TurkDetect Pro - Strict Mode", layout="wide")

# --- OPENAI ANALİZ FONKSİYONU (GÜNCELLENDİ) ---
def extract_names_openai(names_chunk, api_key):
    """
    İsim listesini GPT-4o-mini'ye gönderir.
    Çok sıkı kurallarla SADECE Türkiye Türklerini filtreler.
    """
    client = OpenAI(api_key=api_key)
    
    # --- KRİTİK BÖLÜM: GELİŞTİRİLMİŞ PROMPT ---
    system_prompt = """
    You are a highly strict demographic classifier focused ONLY on identifying people from TURKEY (Turkish Republic context).
    
    YOUR GOAL:
    Filter out names that are strictly Turkish. You must distinguish between "General Islamic/Arabic names" and "Turkish names".

    STRICT RULES FOR VALIDATION:
    1. **Spelling Matters:** - REJECT "Mohammed", "Muhammad", "Ahmad", "Omar". 
       - ACCEPT "Mehmet", "Muhammet", "Ahmet", "Omer".
       - Turks use specific variations (e.g., "Ayse" instead of "Aisha", "Hatice" instead of "Khadija").
    
    2. **Surname Dependency:**
       - If a First Name is common/ambiguous (like "Ali", "Can", "Sara", "Deniz"), the Last Name MUST be undeniably Turkish (e.g., Yilmaz, Ozturk, Kaya, Demir, Sahin).
       - REJECT pairs like "Ali Khan", "Mohammed Asharaf", "Sara Smith".
    
    3. **Exclude Non-Turkish Origins:**
       - Exclude Arab, Persian, Kurdish-only, or Central Asian naming conventions unless they strictly fit the Turkey context.
       - REJECT surnames typically ending in "-ov", "-ev", "-zad", "-zai" unless common in Turkey.
       
    4. **Turkish Surnames:**
       - Look for words with clear Turkish meaning or suffixes: -oglu, -gil, -er, -sen, -soy, -tas, -tepe, -kaya.
       - Common words: Demir, Celik, Yildiz, Yilmaz, Aydin, Arslan.

    Input: A list of full names.
    Output: A JSON object with a key "turkish_names" containing strictly validated Full Names.
    """

    user_prompt = f"""
    Analyze this list. Be extremely selective. Only keep names that look like a native citizen of Turkey:
    {json.dumps(names_chunk)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0 # Sıfır yaratıcılık, tam determinizm.
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("turkish_names", [])

    except Exception as e:
        st.error(f"OpenAI API Hatası: {e}")
        return []

# --- ARAYÜZ (UI) ---
st.title("🇹🇷 TurkDetect | Ultra-Strict Mode")
st.markdown("""
Bu versiyon **GPT-4o-mini** kullanır ve **çok sıkı** bir filtreleme uygular. 
*Mohammed Asharaf* gibi Arapça kökenli isimleri eler, sadece Türkiye formatındaki (*Mehmet, Ahmet*) yazımları ve Türkçe soyisim kombinasyonlarını kabul eder.
""")

# Sidebar
with st.sidebar:
    st.header("🔑 Ayarlar")
    api_key = st.text_input("OpenAI API Key", type="password", help="platform.openai.com adresinden alabilirsiniz.")
    st.info("Bu mod daha seçicidir. Listede azalma olabilir ama doğruluk artar.")
    
    st.markdown("---")
    st.subheader("⚡ Hız Ayarı")
    # Batch size'ı biraz düşürdük ki model daha dikkatli baksın
    batch_size = st.slider("Paket Boyutu", 10, 80, 40, help="Daha düşük sayı = Daha dikkatli analiz.")

# Ana Ekran
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Veri Yükleme")
    uploaded_file = st.file_uploader("CSV Dosyası (Max 50k Satır)", type=["csv"])

    if uploaded_file and api_key:
        df = pd.read_csv(uploaded_file)
        
        # Kolonları Otomatik Bul
        fname_col = next((c for c in df.columns if c.lower() in ['first name', 'firstname', 'ad', 'name']), None)
        lname_col = next((c for c in df.columns if c.lower() in ['last name', 'lastname', 'soyad', 'surname']), None)

        if fname_col and lname_col:
            st.success(f"✅ Dosya doğrulandı: {len(df)} satır.")
            
            # Geçici Tam İsim Kolonu
            df['Full_Name_Temp'] = df[fname_col].astype(str) + " " + df[lname_col].astype(str)
            all_names = df['Full_Name_Temp'].tolist()
            
            if st.button("🚀 Seçici Analizi Başlat"):
                identified_turkish_names = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_batches = (len(all_names) + batch_size - 1) // batch_size
                
                start_time = time.time()
                
                for i in range(0, len(all_names), batch_size):
                    batch = all_names[i : i + batch_size]
                    
                    found = extract_names_openai(batch, api_key)
                    identified_turkish_names.extend(found)
                    
                    # İlerleme
                    current_batch = (i // batch_size) + 1
                    prog = min(current_batch / total_batches, 1.0)
                    progress_bar.progress(prog)
                    status_text.text(f"Taranıyor: {current_batch}/{total_batches} Paket | Bulunan Türk: {len(identified_turkish_names)}")
                    
                    time.sleep(0.1) # API nezaketi

                duration = time.time() - start_time
                st.success(f"İşlem {duration:.2f} saniyede tamamlandı.")
                
                # Sonuçları Filtrele
                turkish_set = set(identified_turkish_names)
                result_df = df[df['Full_Name_Temp'].isin(turkish_set)].copy()
                result_df.drop(columns=['Full_Name_Temp'], inplace=True)
                
                st.session_state['results_strict'] = result_df

        else:
            st.error("CSV dosyasında 'First Name' ve 'Last Name' kolonları bulunamadı.")
    elif uploaded_file and not api_key:
        st.warning("Lütfen OpenAI API anahtarınızı giriniz.")

# Sonuç Ekranı
if 'results_strict' in st.session_state:
    res = st.session_state['results_strict']
    with col2:
        st.subheader("🎯 Filtrelenmiş Sonuçlar")
        st.info(f"Toplam {len(res)} kişi, Türkiye standartlarına göre doğrulandı.")
        st.dataframe(res, height=600)
        
        # Excel İndir
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            res.to_excel(writer, index=False, sheet_name='Strict Turkish Leads')
            
        st.download_button(
            label="📥 Excel İndir (Strict Mode)",
            data=buffer.getvalue(),
            file_name="turkish_leads_strict.xlsx",
            mime="application/vnd.ms-excel"
        )
