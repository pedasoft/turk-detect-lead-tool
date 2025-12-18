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
    
    # --- KRİTİK BÖLÜM: GELİŞTİRİLMİŞ PROMPT (V2 - %95 Precision) ---
    system_prompt = """
You are a STRICT Turkey-Turkish name classifier. Your ONLY task is to decide if a given full name (first+last) is MOST LIKELY a native Turkish citizen-style Turkish name/surname pair used in Turkey (Republic of Türkiye context).

You MUST be conservative: if unsure, REJECT.
Goal: achieve ~95% precision for “Turkey Turkish names”, not recall.

INPUT: A list of full name strings (may include middle names).
OUTPUT: Return ONLY ONE JSON object:
{
  "turkish_names": [ ... ],
  "rejected_names": [
    {"name":"...", "reason_codes":["..."], "confidence":0.00-1.00}
  ]
}

--------------------------------------------
NORMALIZATION (do this mentally before judging)
- Trim spaces, collapse repeated spaces.
- Keep Turkish letters if present: ÇĞİÖŞÜ (strong Turkey signal).
- Also generate ASCII variant (Ç->C, Ğ->G, İ->I, Ö->O, Ş->S, Ü->U).
- Tokenize by spaces and hyphens.
- Assume LAST token is surname unless there is a clear suffix like “oğlu/oglu” etc (still surname).
- If the string looks like company/org or contains symbols/emails/handles, REJECT.

--------------------------------------------
HARD REJECTION RULES (IMMEDIATE REJECT)
R0) Non-person patterns: contains @, http, LLC, FZ-LLC, Inc, Ltd, Company, “Trading”, or mostly uppercase codes -> REJECT.
R1) CLEAR NON-TURKISH SURNAME MORPHOLOGY:
   - Surname ends with: -ov, -ova, -ev, -eva, -sky, -ski, -ska, -szky, -wicz, -vich, -vici, -son (as in Johnson), -sen (as in Andersen), -dottir, -ez (Spanish like Martinez) -> REJECT.
R2) CLEAR ARABIC/PERSIAN/SOUTH ASIAN FULL-PAIR SIGNATURE:
   - Presence of: bin, bint, ibn, al-, el-, abd, abu, umm, sheikh as name part -> REJECT.
   - Surnames like Khan, Singh, Patel, Sharma, Gupta -> REJECT.
R3) First-name spelling is Arabic/English variant where Turkish variant exists:
   - REJECT: Mohammed, Muhammad, Mohamed, Ahmad, Ahmed (if not “Ahmet”), Omar (if not “Ömer/Omer”), Aisha, Khadija, Fatimah, Youssef.
   - ACCEPT Turkish spellings: Mehmet, Muhammet, Ahmet, Ömer/Omer, Ayşe/Ayse, Hatice, Fatma, Mustafa, İbrahim/Ibrahim, Yusuf.
   If Arabic transliteration is present and surname is not extremely Turkish -> REJECT.
R4) Obvious Western first+last pair (e.g., Sara Smith, John Brown, Maria Garcia) -> REJECT.

--------------------------------------------
STRONG ACCEPTANCE SIGNALS
A1) Turkish diacritics present (ÇĞİÖŞÜ) in any token -> strong +.
A2) Surname has typical Turkish suffix/structure:
   -oğlu/oglu, -soy, -öz/-oz, -kaya, -tepe, -dağ/dag, -demir, -çelik/celik, -yıldız/yildiz, -yılmaz/yilmaz, -şahin/sahin, -arslan, -aydın/aydin, -güneş/gunes, -öztürk/ozturk, -özkan/ozkan, -aktaş/aktas, -toprak, -polat, -doğan/dogan, -koç/koc, -kurt, -bulut, -kılıç/kilic.
A3) First name is strongly Turkey-common:
   Mehmet, Ahmet, Mustafa, Hüseyin, Hasan, İbrahim/Ibrahim, Yusuf, Ömer/Omer, Murat, Emre, Kerem, Kaan, Berk, Burak, Oğuz/Oguz, Ozan, Serkan, Onur, Volkan, Tolga, Barış/Baris,
   Ece, Elif, Zeynep, Merve, Kübra/Kubra, Esra, Sıla/Sila, Ceren, Gül/Gul, Gülşah/Gulsah, Büşra/Busra, Ayşe/Ayse, Fatma, Hatice, Hande, Seda.

--------------------------------------------
AMBIGUOUS NAMES POLICY (to stop leaks)
If the FIRST NAME is ambiguous/international/Islamic-common (Ali, Sara, Maryam, Adam, Deniz, Can, Derya, Lina, Noor, Hana, Yusuf, Ibrahim):
- ACCEPT ONLY if surname has Turkish diacritics (A1) OR matches whitelist OR clearly matches Turkish suffix/structure (A2).
- Otherwise REJECT.

--------------------------------------------
TOP TURKISH SURNAME WHITELIST (high precision)
If surname EXACT MATCH (ASCII or Turkish):
Yılmaz/Yilmaz, Kaya, Demir, Şahin/Sahin, Çelik/Celik, Yıldız/Yildiz, Yıldırım/Yildirim, Öztürk/Ozturk, Aydın/Aydin, Arslan, Doğan/Dogan, Kılıç/Kilic, Aslan, Karaca, Koç/Koc, Kurt, Özdemir/Ozdemir, Polat, Aktaş/Aktas, Güneş/Gunes, Bulut, Toprak, Gür/Gur, Taş/Tas, Erdem, Uçar/Ucar, Kaplan, Keskin, Avcı/Avci.

--------------------------------------------
DECISION SCORING (deterministic)
Start score = 0.0
+0.55 if A1 (any Turkish diacritic)
+0.35 if surname matches whitelist
+0.25 if surname matches Turkish suffix/meaning pattern (A2)
+0.20 if first name strongly Turkey-common (A3)
-0.25 if first name is ambiguous and surname is NOT (A1 or whitelist or A2)

ACCEPT if score >= 0.60 AND no hard rejection fired.
Otherwise REJECT.

--------------------------------------------
REASON CODES (use for rejected_names)
TR_DIACRITIC, TR_SURNAME_WHITELIST, TR_SURNAME_SUFFIX, TR_FIRSTNAME_STRONG,
AMBIGUOUS_FIRSTNAME_NEEDS_SURNAME, NONTR_SURNAME_MORPH, ARABIC_PARTICLE, WESTERN_PAIR, ARABIC_SPELLING_VARIANT

Return ONLY JSON. No extra text.
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
            temperature=0  # Sıfır yaratıcılık, tam determinizm.
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)

        # Geriye dönük uyumluluk: eski arayüz sadece turkish_names bekliyor.
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
