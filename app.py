import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import time
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="TurkDetect - GPT-4o-mini", layout="wide")

# --- OPENAI ANALİZ FONKSİYONU ---
def extract_names_openai(names_chunk, api_key):
    """
    İsim listesini GPT-4o-mini'ye gönderir ve Türk olanları JSON formatında alır.
    """
    client = OpenAI(api_key=api_key)
    
    system_prompt = """
    You are a strictly deterministic classifier specialized in demographics.
    Your task is to identify people of TURKISH origin based on their names from a given list.
    
    Rules:
    1. Analyze both First Name and Last Name together for context.
    2. Detect Turkish characters (ş, ğ, ü, ö, ç, ı) even if anglicized (s, g, u, o, c, i).
    3. Look for linguistic roots and suffixes (-oglu, -kaya, -er, -sen, etc.).
    4. Be strict: Exclude common international names unless the surname is distinctly Turkish.
    
    Output Format:
    Return a valid JSON object with a key "turkish_names" containing the array of identified full names.
    """

    user_prompt = f"""
    Analyze this list of names and extract the Turkish ones:
    {json.dumps(names_chunk)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # En hızlı ve maliyet etkin model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # Kesin JSON garantisi
            temperature=0 # Deterministik olması için
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("turkish_names", [])

    except Exception as e:
        st.error(f"OpenAI API Hatası: {e}")
        return []

# --- ARAYÜZ (UI) ---
st.title("🇹🇷 TurkDetect | GPT-4o-mini")
st.markdown("""
Bu araç, OpenAI'nin en hızlı modeli **GPT-4o-mini**'yi kullanarak CSV dosyasındaki 
Türk isimlerini tespit eder.
""")

# Sidebar
with st.sidebar:
    st.header("🔑 Ayarlar")
    api_key = st.text_input("OpenAI API Key", type="password", help="platform.openai.com adresinden alabilirsiniz.")
    st.info("Not: Bu uygulama GPT-4o-mini modelini kullanır. Çok ucuzdur ancak API bakiyeniz olması gerekir.")
    
    st.markdown("---")
    st.subheader("⚡ Hız Ayarı")
    batch_size = st.slider("Paket Boyutu (Batch Size)", 20, 100, 50, help="Tek seferde AI'ya sorulacak isim sayısı.")

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
            
            if st.button("🚀 Analizi Başlat"):
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
                    status_text.text(f"İşleniyor: {current_batch}/{total_batches} Paket | Bulunan: {len(identified_turkish_names)}")
                    
                    # OpenAI çok hızlıdır, rate limit yoksa sleep gerekmez ama güvenlik için:
                    time.sleep(0.1)

                duration = time.time() - start_time
                st.success(f"İşlem {duration:.2f} saniyede tamamlandı.")
                
                # Sonuçları Filtrele
                turkish_set = set(identified_turkish_names)
                result_df = df[df['Full_Name_Temp'].isin(turkish_set)].copy()
                result_df.drop(columns=['Full_Name_Temp'], inplace=True)
                
                st.session_state['results_gpt'] = result_df

        else:
            st.error("CSV dosyasında 'First Name' ve 'Last Name' kolonları bulunamadı.")
    elif uploaded_file and not api_key:
        st.warning("Lütfen OpenAI API anahtarınızı giriniz.")

# Sonuç Ekranı
if 'results_gpt' in st.session_state:
    res = st.session_state['results_gpt']
    with col2:
        st.subheader("🎯 Sonuçlar")
        st.info(f"Toplam {len(res)} Türk kişi bulundu.")
        st.dataframe(res, height=600)
        
        # Excel İndir
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            res.to_excel(writer, index=False, sheet_name='Turkish Leads')
            
        st.download_button(
            label="📥 Excel İndir",
            data=buffer.getvalue(),
            file_name="gpt4o_mini_leads.xlsx",
            mime="application/vnd.ms-excel"
        )
