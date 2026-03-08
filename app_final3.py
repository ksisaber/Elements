import io
import json
import time
import datetime
import requests
import pandas as pd
import streamlit as st

# Configuration page
st.set_page_config(
    page_title="Boussole · Moteur de recherche",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    :root { --dark:#1B3A2D; --green:#2C6E49; --lime:#4CAF50; --gold:#F5C518; --cream:#F5F3EB; }
    .stApp { background-color: var(--cream); }
    [data-testid="stSidebar"] { background-color: var(--dark) !important; }
    [data-testid="stSidebar"] * { color: #e0ede5 !important; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea {
        background-color: #2C4A3A !important; color: #e0ede5 !important;
        border: 1px solid #4CAF50 !important;
    }
    [data-testid="stSidebar"] label { color: #A8C5B0 !important; }
    .stButton > button {
        background-color: var(--lime) !important; color: var(--dark) !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 6px !important; width: 100%;
    }
    .stButton > button:hover { background-color: #43A047 !important; color: white !important; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem 1rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid var(--lime);
    }
    .metric-number { font-size: 2.4rem; font-weight: 800; color: var(--dark); line-height: 1; }
    .metric-label  { font-size: 0.8rem; color: #64748B; margin-top: 0.3rem;
                     text-transform: uppercase; letter-spacing: 0.05em; }
    .domain-badge {
        display: inline-block; background: #E8F5E9; color: var(--green);
        border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    }
    .source-fulltext { background:#E8F5E9; color:#2C6E49; border-radius:4px;
                       padding:1px 7px; font-size:0.72rem; font-weight:700; }
    .source-abstract  { background:#FFF9E6; color:#B8860B; border-radius:4px;
                       padding:1px 7px; font-size:0.72rem; font-weight:700; }
    .effect-positive { color: #2C6E49; font-weight: 700; }
    .effect-negative { color: #C62828; font-weight: 700; }
    .history-card {
        background: #2C4A3A; border-radius: 8px; padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem; border-left: 3px solid #4CAF50;
    }
    .history-title { font-size: 0.85rem; font-weight: 700; color: #E0EDE5; }
    .history-meta  { font-size: 0.72rem; color: #7BA892; margin-top: 2px; }
    hr { border-color: #E8E5DC; }
</style>
""", unsafe_allow_html=True)

# Historique des recherches
if "history" not in st.session_state:
    st.session_state.history = []
if "active_history_idx" not in st.session_state:
    st.session_state.active_history_idx = None
if "selected_preds" not in st.session_state:
    st.session_state.selected_preds = set()

# Helpers claude

def call_claude(prompt: str, system: str = "", max_tokens: int = 2000, api_key: str = "") -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def parse_json(text: str):
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def reconstruct_abstract(idx: dict | None) -> str:
    if not idx:
        return ""
    slots = [""] * (max(p for pp in idx.values() for p in pp) + 1)
    for word, positions in idx.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots)


@st.cache_data
def load_predictors(file_path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_excel(file_path, sheet_name="Predictors")
    df = df[["Predictor (EN)", "Predictor (FR)", "Domain (FR)"]].dropna(subset=["Predictor (EN)"])
    df.columns = ["predictor_en", "predictor_fr", "domain_fr"]
    prompt_str = "\n".join(f"- [{r.domain_fr}] {r.predictor_en}" for _, r in df.iterrows())
    return df, prompt_str

# PIPELINE STEPS

def step1_select_predictors(project, predictors_prompt, top_n, api_key):
    resp = call_claude(
        system="Tu es expert en évaluation d'impact social et modèle Margolis SWB. Réponds UNIQUEMENT en JSON valide.",
        prompt=f"""
Projet : {project['title']}
Groupe : {project['target_group']}
Description : {project['description']}
Tags : {', '.join(project['tags'])}

Voici les 79 prédicteurs du modèle Margolis :
{predictors_prompt}

Sélectionne les {top_n} prédicteurs les plus susceptibles d\'être impactés.
- predictor_en    : nom exact
- domain_fr       : domaine
- relevance_score : 1-5
- justification   : 1 phrase

```json
[{{"predictor_en": "...", "domain_fr": "...", "relevance_score": 5, "justification": "..."}}]
```""",
        max_tokens=1500, api_key=api_key
    )
    return sorted(parse_json(resp), key=lambda x: x.get("relevance_score", 0), reverse=True)


def step2_build_query(pred_name, project, api_key):
    return call_claude(
        prompt=f"""Generate a short academic search query (4-6 words, ENGLISH ONLY) to find
meta-analyses or RCTs measuring the effect size of interventions on: {pred_name}.
The query MUST include the words 'effect size' or 'meta-analysis'.
Respond with ONLY the query, no quotes, no punctuation, no explanation.""",
        max_tokens=30, api_key=api_key
    ).strip().strip('"').strip("'")


def step2_search_openalex(query, n, min_citations, email):
    resp = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": query, "per-page": n, "sort": "relevance_score:desc",
            "select": "title,abstract_inverted_index,doi,publication_year,cited_by_count,open_access,authorships",
            "mailto": email,
        },
        timeout=30
    ).json().get("results", [])
    return [
        {
            "title":   w.get("title", ""),
            "abstract":reconstruct_abstract(w.get("abstract_inverted_index")),
            "doi":     w.get("doi", ""),
            "year":    w.get("publication_year"),
            "cited":   w.get("cited_by_count", 0),
            "oa":      w.get("open_access", {}).get("is_oa", False),
            "authors": ", ".join(
                a["author"]["display_name"]
                for a in w.get("authorships", [])[:2] if a.get("author")
            ),
        }
        for w in resp if w.get("cited_by_count", 0) >= min_citations
    ]


def step3_extract_effect(article, pred_name, project, api_key):
    """Extraction de l'effect size depuis l'abstract OpenAlex."""
    abstract = article.get("abstract", "")[:2000] or "[Abstract non disponible]"
    resp = call_claude(
        system="Tu es expert en méta-analyse. Extrais les tailles d'effet depuis les abstracts. Réponds UNIQUEMENT en JSON valide.",
        prompt=f"""Article : {article['title']} ({article['year']})
Abstract : {abstract}

Prédicteur ciblé : {pred_name}
Projet : {project['title']} | Groupe : {project['target_group']}

Retiens l\'effect size le plus directement lié à {pred_name}.

```json
{{
  "effect_size"     : <float ou null>,
  "effect_type"     : <"Cohen\'s d"|"Hedge\'s g"|"SMD"|"autre"|null>,
  "effect_direction": <"positif"|"négatif"|null>,
  "effect_duration" : <durée pendant laquelle l'effet persiste après l'intervention, ex: "6 months", "1 year", "2 years", "short-term only" — null si non mentionné>,
  "relevance"       : <1-5>,
  "confidence"      : <1-5>,
  "source_text"     : <citation exacte du passage de l'abstract d'où l'effect size a été extrait, ou null si non trouvé>,
  "note"            : <string court>
}}
```""",
        max_tokens=500, api_key=api_key
    )
    try:
        return parse_json(resp)
    except Exception:
        return {"relevance": 1, "confidence": 1, "note": "Erreur d'extraction"}

# SIDEBAR

with st.sidebar:
    st.markdown("## 🧭 Boussole")
    st.markdown("*Moteur de recherche d'effect sizes*")
    st.divider()

    st.markdown("### 🔑 Configuration")
    api_key        = st.text_input("Clé API Anthropic", type="password", placeholder="sk-ant-...")
    openalex_email = st.text_input("Email OpenAlex", placeholder="votre@email.com")

    st.divider()
    st.markdown("### 📋 Projet")
    project_title       = st.text_input("Titre *", placeholder="Ex: Atelier sport & insertion")
    project_description = st.text_area("Description *", placeholder="Actions concrètes...", height=110)
    target_group        = st.text_input("Groupe cible *", placeholder="Ex: Jeunes 16-25 ans")
    tags_raw            = st.text_input("Tags (optionnel)", placeholder="sport, insertion, ...")

    st.divider()
    st.markdown("### 🎯 Mode de sélection")
    selection_mode = st.radio(
        "Comment choisir les prédicteurs ?",
        ["🤖 Automatique (LLM)", "✋ Manuel"],
        help="Automatique : le LLM identifie les prédicteurs pertinents. Manuel : vous les choisissez vous-même."
    )

    st.divider()
    st.markdown("### ⚙️ Paramètres")
    if selection_mode == "🤖 Automatique (LLM)":
        top_n = st.slider("Prédicteurs à explorer",  3, 79, 8)
    articles_per_pred = st.slider("Articles par prédicteur", 1,  20, 3)
    min_citations     = st.slider("Citations minimum",       0, 30, 3)

    st.divider()
    run_button = st.button("🔍 Lancer la recherche", use_container_width=True)

    # Historique
    if st.session_state.history:
        st.divider()
        st.markdown("### 🕓 Historique")
        if st.button("🗑 Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_history_idx = None
            st.rerun()
        for idx, entry in enumerate(reversed(st.session_state.history)):
            real_idx   = len(st.session_state.history) - 1 - idx
            n_effects  = sum(1 for r in entry["results"] if r.get("effect_size") is not None)
            is_active  = st.session_state.active_history_idx == real_idx
            st.markdown(f"""
            <div class="history-card" style="{'border-left-color:#F5C518' if is_active else ''}">
                <div class="history-title">{'▶ ' if is_active else ''}{entry['project']['title'][:36]}{'…' if len(entry['project']['title'])>36 else ''}</div>
                <div class="history-meta">{entry['timestamp']} · {n_effects} effect sizes</div>
            </div>""", unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                if st.button("Voir" if not is_active else "✓ Actif", key=f"h_view_{real_idx}", use_container_width=True):
                    st.session_state.active_history_idx = real_idx
                    st.rerun()
            with c2:
                if st.button("✕", key=f"h_del_{real_idx}", use_container_width=True):
                    st.session_state.history.pop(real_idx)
                    if st.session_state.active_history_idx == real_idx:
                        st.session_state.active_history_idx = None
                    st.rerun()

# Page principale
st.markdown("# 🧭 Moteur de recherche d'effect sizes")

# Chargement des prédicteurs (nécessaire pour mode manuel et LLM)
try:
    df_pred, predictors_prompt = load_predictors("articles_dataset.xlsx")
except FileNotFoundError:
    st.error("❌ `articles_dataset.xlsx` introuvable.")
    st.stop()

# ── Mode Manuel : Sélection des prédicteurs ──────────────────────────────────
manual_selected = []
if selection_mode == "✋ Manuel":
    st.markdown("## 1️⃣ Sélectionnez les prédicteurs à explorer")
    st.caption("Cochez les prédicteurs que votre projet est susceptible d'impacter. Organisés par domaine Margolis.")

    search_pred = st.text_input("🔎 Rechercher un prédicteur", placeholder="Ex: stress, sleep, social...")

    col_all, col_none, _ = st.columns([1, 1, 4])
    with col_all:
        if st.button("✅ Tout sélectionner"):
            st.session_state.selected_preds = set(df_pred["predictor_en"].tolist())
            st.rerun()
    with col_none:
        if st.button("✕ Tout désélectionner"):
            st.session_state.selected_preds = set()
            st.rerun()

    domains = sorted(df_pred["domain_fr"].unique())
    domain_to_preds = {d: df_pred[df_pred["domain_fr"] == d]["predictor_en"].tolist() for d in domains}

    for domain in domains:
        preds_in_domain = domain_to_preds[domain]
        if search_pred:
            preds_in_domain = [p for p in preds_in_domain if search_pred.lower() in p.lower()]
        if not preds_in_domain:
            continue

        n_selected_in_domain = sum(1 for p in preds_in_domain if p in st.session_state.selected_preds)
        with st.expander(
            f"**{domain}** — {len(preds_in_domain)} prédicteurs"
            + (f"  ✅ {n_selected_in_domain} sélectionné(s)" if n_selected_in_domain else ""),
            expanded=(n_selected_in_domain > 0 or bool(search_pred))
        ):
            dc1, dc2 = st.columns([1, 1])
            with dc1:
                if st.button(f"✅ Tout {domain[:15]}", key=f"all_{domain}"):
                    for p in preds_in_domain:
                        st.session_state.selected_preds.add(p)
                    st.rerun()
            with dc2:
                if st.button(f"✕ Aucun {domain[:15]}", key=f"none_{domain}"):
                    for p in preds_in_domain:
                        st.session_state.selected_preds.discard(p)
                    st.rerun()

            cols = st.columns(3)
            for i, pred_name in enumerate(preds_in_domain):
                with cols[i % 3]:
                    pred_fr = df_pred[df_pred["predictor_en"] == pred_name]["predictor_fr"].values
                    checked = pred_name in st.session_state.selected_preds
                    new_val = st.checkbox(pred_name, value=checked, key=f"cb_{pred_name}",
                                          help=f"FR : {pred_fr[0]}" if len(pred_fr) > 0 else "")
                    if new_val != checked:
                        if new_val:
                            st.session_state.selected_preds.add(pred_name)
                        else:
                            st.session_state.selected_preds.discard(pred_name)

    manual_selected = [p for p in df_pred["predictor_en"].tolist() if p in st.session_state.selected_preds]
    st.divider()
    if manual_selected:
        st.markdown(f"**{len(manual_selected)} prédicteur(s) sélectionné(s)**")
    else:
        st.info("Aucun prédicteur sélectionné. Cochez au moins un prédicteur pour lancer la recherche.")
    st.divider()


if not run_button and st.session_state.active_history_idx is None and not st.session_state.history:
    if selection_mode == "🤖 Automatique (LLM)":
        c1, c2, c3 = st.columns(3)
        for col, num, label in [
            (c1, "1", "LLM → Prédicteurs"),
            (c2, "2", "OpenAlex → Articles"),
            (c3, "3", "LLM → Effect sizes"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="metric-number">{num}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("""
**Pipeline en 3 étapes :**
- **Étape 1** — 2 options : soit Le LLM sélectionne les prédicteurs Margolis les plus pertinents pour le projet / Soit on les séléctionne manuellement
- **Étape 2** — OpenAlex recherche des articles académiques pour chaque prédicteur
- **Étape 3** — Le LLM extrait les effect sizes (Cohen's d / Hedge's g) depuis les abstracts
- **Résultat** -- Exportable au format Excel adapté a Boussole
        """)
    st.stop()

if not run_button and st.session_state.active_history_idx is not None:
    entry = st.session_state.history[st.session_state.active_history_idx]
    results             = entry["results"]
    selected_predictors = entry["predictors"]
    project             = entry["project"]
    df = pd.DataFrame(results)
    st.markdown(f"### 🕓 Historique — *{project['title']}*")
    st.caption(f"Recherche du {entry['timestamp']}")
    if st.button("← Nouvelle recherche"):
        st.session_state.active_history_idx = None
        st.rerun()
elif not run_button and not st.session_state.history:
    st.stop()
elif not run_button:
    st.info("👈 Lancez une nouvelle recherche ou consultez l'historique.")
    st.stop()

if run_button:
    errors = []
    if not api_key:             errors.append("🔑 Clé API manquante")
    if not project_title:       errors.append("📋 Titre manquant")
    if not project_description: errors.append("📋 Description manquante")
    if not target_group:        errors.append("📋 Groupe cible manquant")
    if selection_mode == "✋ Manuel" and not manual_selected:
        errors.append("🎯 Aucun prédicteur sélectionné")
    if errors:
        for e in errors: st.error(e)
        st.stop()

    project = {
        "title":        project_title,
        "description":  project_description,
        "target_group": target_group,
        "tags":         [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else [],
    }
    email = openalex_email or "boussole@elements-impact.fr"

    st.markdown(f"### 🔍 Recherche — *{project_title}*")
    results, selected_predictors = [], []

    # ÉTAPE 1 : Sélection des prédicteurs
    if selection_mode == "🤖 Automatique (LLM)":
        with st.status("**Étape 1 — Identification des prédicteurs (LLM)...**", expanded=True) as s1:
            try:
                selected_predictors = step1_select_predictors(project, predictors_prompt, top_n, api_key)
                s1.update(label=f"✅ {len(selected_predictors)} prédicteurs identifiés par le LLM", state="complete", expanded=False)
            except Exception as e:
                s1.update(label=f"❌ {e}", state="error"); st.stop()
    else:
        # Mode manuel : construire la liste de prédicteurs à partir de la sélection utilisateur
        selected_predictors = []
        for pred_name in manual_selected:
            domain = df_pred[df_pred["predictor_en"] == pred_name]["domain_fr"].values
            selected_predictors.append({
                "predictor_en":    pred_name,
                "domain_fr":       domain[0] if len(domain) > 0 else "—",
                "relevance_score": 5,
                "justification":   "Sélectionné manuellement par l'évaluateur",
            })
        st.success(f"✅ {len(selected_predictors)} prédicteur(s) sélectionné(s) manuellement")

    # ÉTAPES 2 & 3 : Recherche + Extraction
    progress = st.progress(0, text="Recherche bibliographique...")

    for i, pred in enumerate(selected_predictors):
        pred_name = pred["predictor_en"]
        progress.progress(i / len(selected_predictors), text=f"🔬 [{i+1}/{len(selected_predictors)}] {pred_name}")

        with st.status(f"**[{i+1}/{len(selected_predictors)}] {pred_name}**", expanded=False):
            # Construction de la requête
            fallback_query = f"{pred_name} intervention effect size meta-analysis"
            if selection_mode == "🤖 Automatique (LLM)":
                try:
                    query = step2_build_query(pred_name, project, api_key)
                except Exception as e:
                    query = fallback_query
            else:
                query = fallback_query
            st.write(f"🔎 `{query}`")

            try:
                articles = step2_search_openalex(query, articles_per_pred, min_citations, email)
                # Fallback : si la requête LLM donne 0 résultats, essayer la requête simple
                if not articles and query != fallback_query:
                    st.write(f"⚠️ 0 résultats → requête de secours : `{fallback_query}`")
                    articles = step2_search_openalex(fallback_query, articles_per_pred, min_citations, email)
                st.write(f"📰 {len(articles)} article(s)")
            except Exception as e:
                st.warning(f"Erreur OpenAlex : {e}"); continue

            for art in articles:
                st.write(f"   📄 *{art['title'][:60]}…*")
                try:
                    extracted = step3_extract_effect(art, pred_name, project, api_key)
                    st.write(f"      effect size : **{extracted.get('effect_size', '—')}**")
                except Exception:
                    extracted = {"relevance": 1, "confidence": 1, "note": "Échec"}

                results.append({
                    "predictor":        pred_name,
                    "domain":           pred["domain_fr"],
                    "pred_score":       pred["relevance_score"],
                    "pred_justif":      pred["justification"],
                    "query":            query,
                    "title":            art["title"],
                    "authors":          art["authors"],
                    "year":             art["year"],
                    "doi":              art["doi"],
                    "cited":            art["cited"],
                    "open_access":      "✅" if art["oa"] else "—",
                    "effect_size":      extracted.get("effect_size"),
                    "effect_type":      extracted.get("effect_type"),
                    "effect_direction": extracted.get("effect_direction"),
                    "effect_duration":  extracted.get("effect_duration"),
                    "relevance":        extracted.get("relevance"),
                    "confidence":       extracted.get("confidence"),
                    "source_text":      extracted.get("source_text", ""),
                    "note":             extracted.get("note", ""),
                })
                time.sleep(0.2)

    progress.progress(1.0, text="✅ Terminé !"); time.sleep(0.5); progress.empty()

    st.session_state.history.append({
        "project":    project,
        "predictors": selected_predictors,
        "results":    results,
        "timestamp":  datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
    })
    st.session_state.active_history_idx = None
    df = pd.DataFrame(results)

# RÉSULTATS
if df.empty:
    st.error("Aucun résultat. Réduisez le filtre de citations.")
    st.stop()

st.divider()
st.markdown("## 📊 Résultats")

n_total       = len(df)
n_with_effect = df["effect_size"].notna().sum()
n_predictors  = df["predictor"].nunique()
pct           = int(n_with_effect / n_total * 100) if n_total > 0 else 0

c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, n_predictors, "Prédicteurs"),
    (c2, n_total,      "Articles"),
    (c3, n_with_effect,"Avec effect size"),
    (c4, f"{pct}%",    "Taux extraction"),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-number">{num}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎯 Prédicteurs", "📄 Articles & Effect sizes", "⬇️ Export"])

with tab1:
    st.markdown("### Prédicteurs identifiés")
    for pred in selected_predictors:
        score = pred["relevance_score"]
        with st.expander(f"{'⭐'*min(score,3)} **{pred['predictor_en']}** — {pred['domain_fr']}"):
            st.markdown(f"*{pred['justification']}*")
            sub = df[df["predictor"] == pred["predictor_en"]]
            with_effect = sub[sub["effect_size"].notna()]
            st.caption(f"{len(sub)} article(s) · {len(with_effect)} effect size(s)")
            if not with_effect.empty:
                best = with_effect.sort_values(["relevance","confidence"], ascending=False).iloc[0]
                es   = float(best["effect_size"])
                st.success(f"**Meilleur :** {best['title'][:75]}…  →  **{es:+.3f}** ({best.get('effect_type','')})")

with tab2:
    st.markdown("### Articles analysés")
    cf1, cf2, cf3 = st.columns(3)
    with cf1: filter_pred   = st.selectbox("Prédicteur", ["Tous"] + sorted(df["predictor"].unique().tolist()))
    with cf2: filter_effect = st.selectbox("Effect size", ["Tous", "Avec uniquement", "Sans uniquement"])
    with cf3: min_rel       = st.selectbox("Pertinence min.", [1,2,3,4,5], index=1)

    df_view = df.copy()
    if filter_pred   != "Tous":            df_view = df_view[df_view["predictor"] == filter_pred]
    if filter_effect == "Avec uniquement": df_view = df_view[df_view["effect_size"].notna()]
    elif filter_effect == "Sans uniquement": df_view = df_view[df_view["effect_size"].isna()]
    df_view = df_view[df_view["relevance"].fillna(0) >= min_rel]
    df_view = df_view.sort_values(["relevance","confidence","cited"], ascending=False)
    st.caption(f"{len(df_view)} article(s)")

    for _, row in df_view.iterrows():
        with st.container():
            cl, cr = st.columns([3, 1])
            with cl:
                doi_link = f"[↗]({row['doi']})" if row["doi"] else ""
                st.markdown(f"**{row['title']}** {doi_link}")
                st.caption(f"{row['authors']}  ·  {row['year']}  ·  {row['cited']} citations  ·  OA: {row['open_access']}")
                st.markdown(f'<span class="domain-badge">{row["domain"]}</span> &nbsp; `{row["predictor"]}`', unsafe_allow_html=True)
                if row.get("note"): st.caption(f"💬 {row['note']}")
            with cr:
                if pd.notna(row.get("effect_size")):
                    es    = float(row["effect_size"])
                    color = "effect-positive" if es >= 0 else "effect-negative"
                    st.markdown(f'<div style="text-align:center"><span class="{color}" style="font-size:1.8rem">{es:+.3f}</span><br><small>{row.get("effect_type","")}</small><br><small style="color:#64748B">{row.get("effect_duration","")}</small></div>', unsafe_allow_html=True)
                    st.caption(f"Pertinence: {row.get('relevance','?')}/5 · Confiance: {row.get('confidence','?')}/5")
                else:
                    st.markdown('<div style="text-align:center;color:#9CA3AF;font-size:0.9rem">Effect size<br>non extrait</div>', unsafe_allow_html=True)
            st.divider()

with tab3:
    st.markdown("### Export")

    # Mapping domaines FR → EN (format Boussole)
    DOMAIN_FR_TO_EN = {
        "Environnement socio-politique":      "Socio-political environment",
        "Environnement matériel & éducation": "Material environment and education",
        "Extra-personnel":                    "Extra-personal",
        "Intra-personnel":                    "Intra-personal",
        "Santé":                              "Health",
        "Travail & Activités":                "Work and Activities",
    }

    def make_boussole_df(df_src: pd.DataFrame, project: dict) -> pd.DataFrame:
        """
        Format Boussole — toutes les lignes (avec ou sans effect size),
        triées par prédicteur puis pertinence/confiance décroissante.
        Colonnes : Project, Project description, Tags, Group, Domain, Predictor,
                   Effect size, Effect duration, Note (raw), Article links

        """
        df_sorted = df_src.sort_values(
            ["predictor", "relevance", "confidence"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        tags_str = ", ".join(project.get("tags", []))
        rows = []
        for _, r in df_sorted.iterrows():
            domain_en = DOMAIN_FR_TO_EN.get(r.get("domain", ""), r.get("domain", ""))

            # ── Note enrichie — passage source de l'effect size ──────────────
            note_parts = []

            # 1. Passage exact d'où l'effect size a été extrait
            if r.get("source_text"):
                note_parts.append(f"Extrait : \"{r['source_text']}\"")

            # 2. Note complémentaire du LLM
            if r.get("note"):
                note_parts.append(r["note"])

            # 3. Source : titre + année + citations
            article_ref = r.get("title", "")
            if r.get("year"):      article_ref += f" ({r['year']})"
            if r.get("cited"):     article_ref += f" — {r['cited']} citations"
            if article_ref:
                note_parts.append(f"Source : {article_ref}")

            # 4. Détails techniques
            details = []
            if r.get("effect_type"):      details.append(f"Type : {r['effect_type']}")
            if r.get("effect_direction"): details.append(f"Direction : {r['effect_direction']}")
            if r.get("confidence"):       details.append(f"Confiance LLM : {r['confidence']}/5")
            if details:
                note_parts.append(" | ".join(details))

            rows.append({
                "Project":             project.get("title", ""),
                "Project description": project.get("description", ""),
                "Tags":                tags_str,
                "Group":               project.get("target_group", ""),
                "Domain":              domain_en,
                "Predictor":           r["predictor"],
                "Effect size":         r["effect_size"],
                "Effect duration":     r.get("effect_duration", ""),
                "Note (raw)":          " ; ".join(note_parts),
                "Article links":       r.get("doi", ""),
            })
        return pd.DataFrame(rows)

    # ── Construction ──────────────────────────────────────────────────────────
    boussole_with    = make_boussole_df(df[df["effect_size"].notna()], project)
    boussole_without = make_boussole_df(df[df["effect_size"].isna()],  project)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Onglet 1 : articles avec effect size — prêt à injecter dans Boussole
        boussole_with.to_excel(writer, sheet_name="Boussole — avec effect size", index=False)
        # Onglet 2 : articles sans effect size — à compléter manuellement
        boussole_without.to_excel(writer, sheet_name="Boussole — sans effect size", index=False)
        # Onglet 3 : tous les articles avec colonnes techniques complètes
        df.sort_values(["relevance", "confidence"], ascending=False).to_excel(
            writer, sheet_name="Tous les articles", index=False)
        # Onglet 4 : prédicteurs sélectionnés
        pd.DataFrame(selected_predictors).to_excel(writer, sheet_name="Prédicteurs", index=False)

    proj_name = project_title[:30].replace(" ", "_") if run_button else "historique"
    st.download_button(
        "⬇️ Télécharger Excel (4 onglets)",
        buffer.getvalue(),
        f"boussole_{proj_name}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    # ── Aperçu ────────────────────────────────────────────────────────────────
    tab_with, tab_without = st.tabs([
        f"✅ Avec effect size ({len(boussole_with)})",
        f"❌ Sans effect size ({len(boussole_without)})",
    ])
    with tab_with:
        if not boussole_with.empty:
            st.caption("Triés par prédicteur puis pertinence — l'évaluateur choisit quelle ligne retenir")
            st.dataframe(boussole_with, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun effect size extrait.")
    with tab_without:
        if not boussole_without.empty:
            st.caption("Articles trouvés mais sans effect size détecté — à compléter manuellement si besoin")
            st.dataframe(boussole_without, use_container_width=True, hide_index=True)
        else:
            st.success("Tous les articles ont un effect size extrait !")

    st.caption("💡 Colonnes identiques à l'onglet *Notes export* de Boussole.")
