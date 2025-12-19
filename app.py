import streamlit as st
import pandas as pd
import requests
import json
import io
from openai import OpenAI

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Sales Hunter", page_icon="🎯", layout="wide")

st.title("🎯 B2B Sales Lead Generator")
st.markdown("Hedef kitlenizi arayın, profilleri ayrıştırın, e-postaları zenginleştirin ve Excel'e aktarın.")

# --- SIDEBAR: AYARLAR ---
with st.sidebar:
    st.header("⚙️ Konfigürasyon")
    
    st.subheader("1. API Anahtarları")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    serper_api_key = st.text_input("Serper (Google) API Key", type="password")
    hunter_api_key = st.text_input("Hunter.io API Key (Opsiyonel)", type="password", help="Gerçek e-posta bulmak için gereklidir. Girilmezse tahmini mail üretilir.")
    
    st.divider()
    
    st.subheader("2. Hedef Kitle")
    target_position = st.text_input("Ünvan", "General Manager")
    target_industry = st.text_input("Sektör", "Construction")
    target_location = st.text_input("Lokasyon", "Istanbul")
    
    search_limit = st.slider("Sonuç Sayısı", 5, 20, 10)

# --- YARDIMCI FONKSİYONLAR ---

def google_search(position, industry, location, api_key, num_results):
    """Google Serper API ile arama yapar."""
    url = "https://google.serper.dev/search"
    # LinkedIn X-Ray Arama Sorgusu
    query = f'site:linkedin.com/in/ "{position}" "{industry}" "{location}"'
    
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def parse_profile(item):
    """
    LinkedIn başlığını (Title) analiz edip Ad, Ünvan ve Şirket bilgisini ayıklar.
    Örnek Title: "Ahmet Yılmaz - Genel Müdür - ABC İnşaat | LinkedIn"
    """
    title = item.get("title", "")
    parts = title.split("-")
    
    # Varsayılan değerler
    name = "Bilinmiyor"
    role = "Bilinmiyor"
    company = "Bilinmiyor"
    
    if len(parts) >= 1:
        name = parts[0].strip()
    if len(parts) >= 2:
        role = parts[1].strip()
    if len(parts) >= 3:
        # Şirket adındaki gereksiz " | LinkedIn" kısmını temizle
        company = parts[2].split("|")[0].strip()
        
    return name, role, company

def find_email_hunter(name, company, api_key):
    """
    Hunter.io API kullanarak mail bulur. 
    Eğer API Key yoksa veya domain bulunamazsa 'pattern' tabanlı tahmin yapar.
    """
    if not api_key:
        # API Key yoksa simülasyon yap (Tahmini format)
        domain = company.lower().replace(" ", "") + ".com"
        email = f"{name.lower().replace(' ', '.')}@{domain}"
        return email, "Tahmini (API Yok)"
    
    # 1. Önce Şirketin Domainini Bulmaya Çalış (Hunter Domain Search)
    domain_url = f"https://api.hunter.io/v2/domain-search?company={company}&api_key={api_key}"
    try:
        domain_res = requests.get(domain_url).json()
        if "data" in domain_res and domain_res["data"].get("domain"):
            domain = domain_res["data"]["domain"]
            
            # 2. Kişinin Mailini Bul (Email Finder)
            # İsim soyisim ayrıştırma
            name_parts = name.split(" ")
            first_name = name_parts[0]
            last_name = name_parts[-1] if len(name_parts) > 1 else ""
            
            finder_url = f"https://api.hunter.io/v2/email-finder?domain={domain}&first_name={first_name}&last_name={last_name}&api_key={api_key}"
            email_res = requests.get(finder_url).json()
            
            if "data" in email_res and email_res["data"].get("email"):
                return email_res["data"]["email"], "Doğrulanmış (Hunter)"
            else:
                return f"Bulunamadı (@{domain})", "Domain bulundu, Kişi bulunamadı"
        else:
            return "Domain Bulunamadı", "Başarısız"
            
    except Exception:
        return "Hata", "API Hatası"

# --- ANA UYGULAMA MANTIĞI ---

def run_app():
    if not serper_api_key:
        st.warning("⚠️ Lütfen sol menüden Serper API anahtarını girin.")
        return

    # Başlatma Butonu
    if st.button("🚀 Taramayı Başlat", type="primary"):
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # 1. ADIM: ARAMA
        status_text.text("🔍 Google üzerinde LinkedIn profilleri taranıyor...")
        results = google_search(target_position, target_industry, target_location, serper_api_key, search_limit)
        progress_bar.progress(30)
        
        if "organic" not in results:
            st.error("Sonuç bulunamadı veya API hatası.")
            return

        items = results["organic"]
        processed_data = []
        
        # 2. ADIM: PARSING VE ENRICHMENT
        status_text.text(f"🧩 {len(items)} profil ayrıştırılıyor ve e-postalar zenginleştiriliyor...")
        
        total_items = len(items)
        for i, item in enumerate(items):
            # Parsing
            name, role, company = parse_profile(item)
            linkedin_url = item.get("link")
            snippet = item.get("snippet")
            
            # Enrichment (Email Bulma)
            email, status = find_email_hunter(name, company, hunter_api_key)
            
            processed_data.append({
                "Ad Soyad": name,
                "Ünvan": role,
                "Şirket": company,
                "E-Posta": email,
                "Durum": status,
                "LinkedIn URL": linkedin_url,
                "Bağlam (Snippet)": snippet
            })
            
            # Progress bar güncelle
            current_progress = 30 + int((i / total_items) * 60)
            progress_bar.progress(current_progress)
            
        progress_bar.progress(100)
        status_text.text("✅ İşlem tamamlandı!")
        
        # 3. ADIM: DATAFRAME OLUŞTURMA
        df = pd.DataFrame(processed_data)
        
        # Ekrana Grid Olarak Basma (Data Editor ile düzenlenebilir yaparız)
        st.subheader("📋 Sonuç Listesi")
        edited_df = st.data_editor(
            df,
            column_config={
                "LinkedIn URL": st.column_config.LinkColumn("Profil Linki"),
                "E-Posta": st.column_config.TextColumn("E-Posta Adresi", help="Otomatik bulunan veya tahmin edilen adres")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 4. ADIM: EXCEL İNDİRME
        st.subheader("💾 Dışa Aktar")
        
        # Excel'i hafızada (RAM) oluşturuyoruz, diske yazmıyoruz (Cloud uyumlu)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Leads')
        
        processed_data = output.getvalue()
        
        st.download_button(
            label="📥 Excel Olarak İndir (.xlsx)",
            data=processed_data,
            file_name=f"leads_{target_industry}_{target_location}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 5. ADIM: AI ANALİZ (Opsiyonel Eklenti)
        if openai_api_key and not df.empty:
            st.divider()
            if st.button("🧠 AI Analizi Yap (İlk 3 Kişi)"):
                client = OpenAI(api_key=openai_api_key)
                st.write("GPT-4 profilleri analiz ediyor...")
                
                for index, row in df.head(3).iterrows():
                    with st.expander(f"Analiz: {row['Ad Soyad']} - {row['Şirket']}"):
                        prompt = f"Şu kişiye satış yapmak istiyorum: {row['Ad Soyad']}, {row['Ünvan']}, {row['Şirket']}. Hakkındaki kısa bilgi: {row['Bağlam (Snippet)']}. Bana bu kişiye atılacak 'hook' (kanca) cümlesini yaz."
                        res = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": prompt}])
                        st.write(res.choices[0].message.content)

if __name__ == "__main__":
    run_app()
