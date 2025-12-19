import streamlit as st
import pandas as pd
import requests
import json
import io
from urllib.parse import urlparse
from openai import OpenAI

# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(
    page_title="AI Sales Hunter (Domain + Apollo Match)",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 B2B Sales Agent: Domain Discovery + Apollo Match")
st.markdown("1) Google/Serper ile LinkedIn bul → 2) GPT ile kişi/şirket ayıkla → 3) Domain bul → 4) Apollo Match ile email enrich")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Konfigürasyon")

    st.subheader("1) API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    serper_api_key = st.text_input("Serper (Google) API Key", type="password")
    apollo_api_key = st.text_input("Apollo.io API Key", type="password")

    st.divider()

    st.subheader("2) Hedef Kitle")
    target_position = st.text_input("Ünvan", "Quality Assurance Manager")
    target_industry = st.text_input("Sektör", "Pharma")
    target_location = st.text_input("Lokasyon", "Dubai")
    search_limit = st.slider("Sonuç Sayısı", 5, 20, 10)

    st.divider()

    st.subheader("3) Apollo Reveal Ayarları")
    reveal_personal_emails = st.toggle("Kişisel emailleri reveal etmeyi dene", value=False)
    reveal_phone_number = st.toggle("Telefon reveal etmeyi dene", value=False)

# -----------------------------
# HELPERS
# -----------------------------
SERPER_URL = "https://google.serper.dev/search"
APOLLO_MATCH_URL = "https://api.apollo.io/api/v1/people/match"


def safe_post(url: str, headers: dict, payload: dict | None = None, params: dict | None = None, timeout: int = 30):
    """requests.post wrapper with basic error handling."""
    r = requests.post(url, headers=headers, json=payload, params=params, timeout=timeout)
    # Apollo bazen non-200 döner, burada body'yi yakalamak faydalı
    try:
        data = r.json()
    except Exception:
        data = {"raw": r.text}
    return r.status_code, data


def google_search_linkedin(position: str, industry: str, location: str, api_key: str, num_results: int):
    """Serper üzerinden Google'da LinkedIn profilleri arar."""
    query = f'site:linkedin.com/in/ "{position}" "{industry}" "{location}"'
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    try:
        status, data = safe_post(SERPER_URL, headers=headers, payload=payload, timeout=30)
        if status >= 400:
            return {"error": f"Serper HTTP {status}", "details": data}
        return data
    except Exception as e:
        return {"error": str(e)}


def clean_domain_from_url(link: str) -> str | None:
    """https://www.nestle.com/jobs -> nestle.com"""
    if not link:
        return None
    try:
        parsed = urlparse(link)
        if not parsed.netloc:
            return None
        return parsed.netloc.replace("www.", "")
    except Exception:
        return None


def find_company_domain(company_name: str, serper_key: str) -> str | None:
    """Şirket isminden domain bulur (serper)."""
    if not company_name or company_name == "Bilinmiyor":
        return None

    query = f'{company_name} official website'
    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 3}

    try:
        status, data = safe_post(SERPER_URL, headers=headers, payload=payload, timeout=30)
        if status >= 400:
            return None
        organic = data.get("organic", [])
        if not organic:
            return None

        # İlk sonucu al ama istersen birkaçını kontrol edip en iyi domaini seçebilirsin
        link = organic[0].get("link", "")
        return clean_domain_from_url(link)
    except Exception:
        return None


def extract_info_with_gpt(raw_title: str, snippet: str, client: OpenAI) -> dict:
    """Google sonucundan kişi / ünvan / şirket ayıklar."""
    prompt = f"""
Aşağıdaki veriden Kişi, Ünvan ve Şirket bilgisini çıkar.

GİRDİ:
Title: {raw_title}
Snippet: {snippet}

KURALLAR:
- Sadece JSON döndür.
- Şirketi mümkünse "at" / "@" / "-" gibi ayırıcılardan yakala.
- Bulamazsan "Bilinmiyor" yaz.

JSON:
{{
  "name": "Ad Soyad",
  "role": "Ünvan",
  "company": "Şirket Adı"
}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"name": "Bilinmiyor", "role": "Bilinmiyor", "company": "Bilinmiyor"}


def apollo_people_match(api_key: str, linkedin_url: str | None, name: str | None, domain: str | None,
                       reveal_personal: bool, reveal_phone: bool):
    """
    Apollo match:
    1) linkedin_url ile dene
    2) name+domain ile dene
    """
    if not api_key:
        return "API Key Yok", "❌ Apollo API Key yok"

    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "X-Api-Key": api_key
    }

    params = {
        "reveal_personal_emails": str(reveal_personal).lower(),
        "reveal_phone_number": str(reveal_phone).lower()
    }

    # 1) LinkedIn match
    if linkedin_url:
        payload = {"linkedin_url": linkedin_url}
        try:
            status, data = safe_post(APOLLO_MATCH_URL, headers=headers, payload=payload, params=params, timeout=30)
            if status < 400:
                person = data.get("person")
                if person:
                    email = person.get("email")
                    if email:
                        return email, "✅ Eşleşti (LinkedIn)"
                    return "Mail Yok", "⚠️ Profil Var, Mail Yok"
            else:
                # Apollo hata döndürürse görmek için aşağıda status'a yazarız
                pass
        except Exception:
            pass

    # 2) Name + domain match
    if not (name and domain):
        if not domain:
            return "Domain Yok", "❌ Domain bulunamadı"
        return "İsim Yok", "❌ İsim parse edilemedi"

    parts = name.split()
    first_name = parts[0] if parts else ""
    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

    payload = {
        "first_name": first_name,
        "last_name": last_name,
        "domain": domain  # <-- doğru alan
    }

    try:
        status, data = safe_post(APOLLO_MATCH_URL, headers=headers, payload=payload, params=params, timeout=30)
        if status >= 400:
            return "Hata", f"API Hatası (HTTP {status})"
        person = data.get("person")
        if person:
            email = person.get("email")
            if email:
                return email, "✅ Eşleşti (Name+Domain)"
            return "Mail Yok", "⚠️ Profil Var, Mail Yok"
        return "Bulunamadı", "❌ Eşleşme Yok"
    except Exception as e:
        return "Hata", f"API Hatası: {str(e)}"


# -----------------------------
# APP
# -----------------------------
def run_app():
    if not (openai_api_key and serper_api_key and apollo_api_key):
        st.warning("⚠️ OpenAI + Serper + Apollo API key'lerini gir.")
        return

    if st.button("🚀 Taramayı Başlat", type="primary"):
        client = OpenAI(api_key=openai_api_key)
        status_box = st.status("İşlem başlıyor...", expanded=True)

        # 1) Google/Serper search
        status_box.write("🔍 Google/Serper ile LinkedIn profilleri aranıyor...")
        results = google_search_linkedin(
            target_position,
            target_industry,
            target_location,
            serper_api_key,
            search_limit
        )

        if "error" in results:
            status_box.update(label="Hata!", state="error")
            st.error(results["error"])
            if "details" in results:
                st.json(results["details"])
            return

        organic = results.get("organic", [])
        if not organic:
            status_box.update(label="Hata!", state="error")
            st.error("Google sonuçları boş döndü.")
            return

        processed = []
        total = len(organic)
        progress = status_box.progress(0)

        for i, item in enumerate(organic, start=1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            linkedin_url = item.get("link")

            status_box.write(f"🧠 Analiz: {i}/{total}")

            # 2) GPT parse
            parsed = extract_info_with_gpt(title, snippet, client)
            name = parsed.get("name", "Bilinmiyor")
            role = parsed.get("role", "Bilinmiyor")
            company = parsed.get("company", "Bilinmiyor")

            # 3) Domain
            domain = find_company_domain(company, serper_api_key) if company != "Bilinmiyor" else None

            # 4) Apollo match
            email, apollo_status = apollo_people_match(
                api_key=apollo_api_key,
                linkedin_url=linkedin_url,
                name=name if name != "Bilinmiyor" else None,
                domain=domain,
                reveal_personal=reveal_personal_emails,
                reveal_phone=reveal_phone_number
            )

            processed.append({
                "Ad Soyad": name,
                "Ünvan": role,
                "Şirket": company,
                "Domain": domain or "",
                "E-Posta": email,
                "Durum": apollo_status,
                "LinkedIn URL": linkedin_url or ""
            })

            progress.progress(i / total)

        status_box.update(label="✅ Tamamlandı!", state="complete", expanded=False)

        df = pd.DataFrame(processed)
        st.subheader(f"📋 Sonuçlar ({len(df)} Kayıt)")

        edited_df = st.data_editor(
            df,
            column_config={
                "LinkedIn URL": st.column_config.LinkColumn("Profil"),
            },
            hide_index=True,
            use_container_width=True
        )

        # Excel Export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            edited_df.to_excel(writer, index=False, sheet_name="Leads")

        st.download_button(
            label="📥 Excel İndir",
            data=output.getvalue(),
            file_name="Leads_Apollo_Match.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )


if __name__ == "__main__":
    run_app()
