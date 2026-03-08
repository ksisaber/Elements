# Moteur de recherche d'articles scientifiques — Elements Impact

Prototype de moteur de recherche automatisé d'**effect sizes** (Cohen's *d* / Hedge's *g*) pour alimenter l'outil **Boussole** d'évaluation d'impact sur le bien-être subjectif (modèle Margolis SWB).

---

## Contexte

Boussole calcule l'impact d'un projet social sur le bien-être subjectif en s'appuyant sur le modèle de régression de Margolis et al., qui recense **79 prédicteurs** répartis en 6 domaines. Pour chaque prédicteur impacté par un projet, un évaluateur doit trouver manuellement une taille d'effet dans la littérature scientifique — un travail chronophage et difficile à standardiser.

Ce moteur automatise cette recherche.

---

## Pipeline

```
Titre + Description du projet + Groupe ou Groupes bénéficiaires du projet + tags
        │
        ▼
  ÉTAPE 1 — Sélection des prédicteurs
  Le LLM identifie les prédicteurs Margolis
  les plus susceptibles d'être impactés
  (score de pertinence 1-5 + justification)
        │
        ▼
  ÉTAPE 2 — Recherche OpenAlex
  Pour chaque prédicteur, le LLM génère
  une requête académique ciblée (anglais,
  "effect size" ou "meta-analysis" obligatoire)
  → fallback automatique si 0 résultats
        │
        ▼
  ÉTAPE 3 — Extraction LLM
  Lecture de l'abstract, extraction du
  Cohen's d / Hedge's g / SMD + durée
  + passage source + score de confiance
        │
        ▼
  Output structuré au format Boussole
  (colonnes identiques à Notes export)
```

---

## Livrables

| Fichier | Description |
|---|---|
| `elements_impact_pipeline.ipynb` | Notebook Jupyter — pipeline complet, 1 cellule à modifier |
| `app_final3.py` | Interface Streamlit — mode LLM automatique + mode manuel |
| `articles_dataset.xlsx` | Fichier de référence — onglet `Predictors` requis |

---

## Installation

```bash
pip install requests pandas openpyxl streamlit
```

> Aucune autre dépendance. Les deux APIs utilisées (Anthropic et OpenAlex) sont appelées directement via `requests`.

---

## Utilisation

### Notebook Jupyter

Ouvrir `elements_impact_pipeline.ipynb` et modifier uniquement la **cellule 2 — Configuration** :

```python
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")  # clé API Anthropic
OPENALEX_EMAIL         = "votre@email.com"                   # améliore le rate limit
TOP_N_PREDICTORS       = 8    # nb de prédicteurs à explorer
ARTICLES_PER_PREDICTOR = 3    # nb d'articles par prédicteur
MIN_CITATIONS          = 3    # filtre qualité (0 = désactivé)
```

Et la **cellule 4 — Input projet** :

```python
PROJECT = {
    "title":        "Nom du projet",
    "description":  "Description des actions menées...",
    "target_group": "Groupe bénéficiaire",
    "tags":         ["tag1", "tag2"],
}
```

Exécuter toutes les cellules dans l'ordre. L'export Excel est généré automatiquement.

### Interface Streamlit

```bash
streamlit run app_final3.py
```

> `articles_dataset.xlsx` doit être dans le même dossier que `app.py`.

L'interface propose deux modes de sélection des prédicteurs :

- **🤖 Automatique (LLM)** — le modèle sélectionne les prédicteurs pertinents à partir du titre, de la description et des tags du projet
- **✋ Manuel** — l'évaluateur choisit lui-même les prédicteurs via une interface de sélection organisée par domaine Margolis, avec barre de recherche filtrante

---

## Output

### Colonnes du DataFrame de résultats

| Colonne | Description |
|---|---|
| `predictor` | Nom du prédicteur Margolis (EN) |
| `domain` | Domaine Margolis (FR) |
| `pred_score` | Score de pertinence LLM (1-5) |
| `pred_justif` | Justification de la sélection |
| `search_query` | Requête envoyée à OpenAlex |
| `title` | Titre de l'article |
| `authors` | Auteurs (2 premiers) |
| `year` | Année de publication |
| `doi` | DOI de l'article |
| `cited` | Nombre de citations |
| `open_access` | Article en open access ? |
| `effect_size` | Taille d'effet extraite (float) |
| `effect_type` | Type : Cohen's d / Hedge's g / SMD |
| `effect_direction` | Direction : positif / négatif |
| `effect_duration` | Durée de persistance de l'effet |
| `relevance` | Pertinence de l'article (1-5) |
| `confidence` | Confiance de l'extraction (1-5) |
| `source_text` | Passage exact de l'abstract cité |
| `note` | Le passage exact de l’abstract à partir duquel l’effect size est extrait, ou, à défaut, une note complémentaire fournie par le LLM.|

### Export Excel — 4 onglets

| Onglet | Contenu |
|---|---|
| `Boussole — avec effect size` | Format exact de l'onglet *Notes export*, toutes les lignes avec une valeur extraite — l'évaluateur choisit quelle ligne retenir |
| `Boussole — sans effect size` | Même format, articles sans effect size trouvé — à compléter manuellement |
| `Tous les articles` | Toutes les colonnes techniques |
| `Prédicteurs` | Prédicteurs sélectionnés avec scores |

L'onglet **Boussole** respecte exactement les colonnes du fichier *Notes export* fourni par Elements Impact : `Project`, `Project description`, `Tags`, `Group`, `Domain`, `Predictor`, `Effect size`, `Effect duration`, `Note (raw)`, `Article links`.

---

## Choix techniques

**Pourquoi OpenAlex ?**
API gratuite, sans clé, 250M+ publications. Fournit les abstracts dans un format index inversé `{mot: [positions]}` (contrainte de droits d'auteur) que le moteur reconstitue automatiquement. Tri par `relevance_score` natif.

**Pourquoi un LLM pour la requête (étape 2) ?**
Un matching par mots-clés échouerait sur les synonymes sémantiques : `"sport"` ne matche pas `"Physical activity"`, `"parentalité"` ne matche pas `"Social integration"`. Le LLM génère une requête académique en anglais forcée à inclure `"effect size"` ou `"meta-analysis"` pour cibler des articles avec des données quantitatives.

**Fallback automatique**
Si la requête LLM renvoie 0 résultats après filtre citations, le moteur retombe automatiquement sur `"{prédicteur} intervention effect size meta-analysis"`.

**Pourquoi un LLM pour l'extraction (étape 3) ?**
Les effect sizes dans les abstracts sont exprimés de façons très hétérogènes (`d = 0.41`, `Hedges' g of 0.3`, `standardized mean difference (SMD) = 0.52`). Le LLM comprend le contexte et sélectionne la valeur la plus directement liée au prédicteur ciblé. Il retourne aussi le `source_text` — le passage exact de l'abstract d'où vient la valeur, traçable et vérifiable par l'évaluateur.

---

## Limites connues

- L'extraction repose sur les **abstracts uniquement** — les effect sizes reportés uniquement dans les tableaux ou sections "Results" ne sont pas capturés
- OpenAlex couvre ~80% du corpus académique — certains articles spécialisés peuvent être absents
- L'extraction LLM est **probabiliste** : le score `confidence` (1-5) permet d'identifier les extractions incertaines à valider manuellement
- Le moteur propose, **l'évaluateur valide** — l'onglet "Boussole — avec effect size" liste toutes les options trouvées pour que l'évaluateur choisisse la plus pertinente
