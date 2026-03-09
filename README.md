# Moteur de recherche d'articles scientifiques — Elements Impact

Prototype de moteur de recherche automatisé d'**effect sizes** (Cohen's *d* / Hedge's *g*) pour alimenter l'outil **Boussole** d'évaluation d'impact sur le bien-être subjectif (modèle Margolis SWB).

---

## Contexte

Boussole calcule l'impact d'un projet social sur le bien-être subjectif en s'appuyant sur le modèle de régression de Margolis et al., qui recense **79 prédicteurs** répartis en 6 domaines. Pour chaque prédicteur impacté par un projet, un évaluateur doit trouver manuellement une taille d'effet dans la littérature scientifique — un travail chronophage, difficile à standardiser, et variable selon l'évaluateur.

Ce moteur automatise cette recherche.

---

## Pipeline

```
Titre + Description + Groupe bénéficiaire + Tags + Contexte de recherche (optionnel)
        │
        ▼
  ÉTAPE 1 — Sélection des prédicteurs (LLM)
  Le LLM reçoit les 79 prédicteurs Margolis et
  identifie ceux que le projet est susceptible
  d'impacter. Chaque prédicteur retenu reçoit
  un score de pertinence (1-5) et une justification.
        │
        ▼  (pour chaque prédicteur)
  ÉTAPE 2a — Génération des requêtes (déterministe)
  Jusqu'à 10 requêtes académiques construites
  mécaniquement — zéro token LLM, reproductible.
    • Q1-Q5 : avec "meta-analysis" / "systematic review"
      → recall maximal sur les méta-analyses
    • Q6-Q10 : sans — élargissent vers les études
      primaires, RCTs et interventions
  Si un contexte de recherche est renseigné
  (ex : "parenting support"), il est injecté
  dans Q6-Q10 pour ancrer la recherche dans
  le domaine spécifique du projet.
        │
        ▼
  ÉTAPE 2b — Recherche OpenAlex
  Chaque requête est soumise à OpenAlex.
  On retient le top 5 par requête (triés par
  relevance_score natif), dédupliqués par DOI.
  Pool résultant : ~50 articles par prédicteur.
        │
        ▼
  ÉTAPE 2c — Sélection LLM
  Le LLM reçoit titres + débuts d'abstract des
  ~50 candidats et sélectionne les N meilleurs
  selon trois critères (par ordre de priorité) :
    1. Compatibilité population / contexte projet
    2. Lien direct avec le prédicteur étudié
    3. Présence probable d'un effect size quantitatif
  → Un seul appel API, max_tokens=60
        │
        ▼
  ÉTAPE 3 — Extraction LLM
  Le LLM lit l'abstract de chaque article retenu
  et extrait : Cohen's d / Hedge's g / SMD + durée
  + passage source exact + score de confiance
  + justification du choix de l'article
        │
        ▼
  Output structuré au format Boussole
  (colonnes identiques à Notes export)
```

---

## Livrables

| Fichier | Description |
|---|---|
| `processus_pipeline.ipynb` | Notebook Jupyter — pipeline complet, 2 cellules à modifier |
| `app.py` | Interface Streamlit — mode LLM automatique + mode manuel |
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

Ouvrir `processus_pipeline.ipynb` et modifier uniquement la **cellule 2 — Configuration** :

```python
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")  # clé API Anthropic
OPENALEX_EMAIL         = "votre@email.com"   # améliore le rate limit OpenAlex
TOP_N_PREDICTORS       = 8    # nb de prédicteurs explorés (étape 1)
ARTICLES_PER_PREDICTOR = 3    # nb d'articles retenus par prédicteur (après sélection LLM)
MIN_CITATIONS          = 3    # filtre qualité — articles avec moins de N citations exclus (0 = désactivé)
OPENALEX_FETCH         = 20   # articles récupérés par requête avant sélection top 5
TOP_PER_QUERY          = 5    # top N articles retenus par requête pour constituer le pool
```

Et la **cellule 4 — Input projet** :

```python
PROJECT = {
    "title":          "Early childhood support and parenting center",
    "description":    "Management of a welcoming space dedicated to parenting support...",
    "target_group":   "Children aged 0-6 and their parents",
    "tags":           ["early childhood", "parenting", "child development", "social support"],
    "search_context": "parenting support",   # optionnel — voir section dédiée ci-dessous
}
```

> **Recommandation importante** : renseigner le projet en **anglais**. Les requêtes étant construites par tokenisation des inputs et envoyées à OpenAlex en anglais, des inputs en français produisent des termes peu utiles et des requêtes moins efficaces.

Exécuter toutes les cellules dans l'ordre. L'export Excel est généré automatiquement à la fin.

### Interface Streamlit

```bash
streamlit run app.py
```

> `articles_dataset.xlsx` doit être dans le même dossier que `app.py`.

L'interface propose deux modes de sélection des prédicteurs :

- **🤖 Automatique (LLM)** — le modèle sélectionne les prédicteurs pertinents à partir du titre, de la description et des tags du projet
- **✋ Manuel** — l'évaluateur choisit lui-même les prédicteurs via une interface de sélection organisée par domaine Margolis, avec barre de recherche filtrante

Dans les deux modes, la génération des requêtes et la sélection des articles suivent exactement la même logique.

---

## Le champ `search_context`

C'est l'input optionnel le plus impactant. Il prend **2 à 3 mots en anglais** décrivant le domaine spécifique du projet — idéalement des termes que l'on retrouverait dans des titres d'articles académiques.

```python
"search_context": "parenting support"      # projet parentalité
"search_context": "workplace burnout"      # programme anti-stress en entreprise
"search_context": "sport adolescents"      # club de sport pour jeunes
"search_context": "cognitive aging"        # activités pour seniors
"search_context": ""                       # laisser vide = recherche générique
```

**Pourquoi c'est utile**

Sans `search_context`, les requêtes Q6-Q10 utilisent les termes extraits automatiquement du titre et du groupe cible — ce qui fonctionne mais peut manquer de précision. Avec `search_context`, ces mêmes requêtes deviennent directement ancrées dans le domaine :

```
# Prédicteur : family support — search_context : "parenting support"

Sans contexte  →  Q6 : "family support children aged wellbeing"
Avec contexte  →  Q6 : "family support parenting wellbeing"

Sans contexte  →  Q7 : "family support intervention children aged"
Avec contexte  →  Q7 : "family support parenting intervention"
```

Les requêtes Q1-Q5 (avec `meta-analysis` / `systematic review`) restent volontairement génériques pour ne pas réduire le recall sur les méta-analyses disponibles.

**Pourquoi ne pas l'extraire automatiquement**

Une extraction automatique depuis les tags ou le titre produit des termes parfois trop génériques (`early`, `mental`) ou trop liés à la description opérationnelle du projet plutôt qu'à la littérature académique. L'utilisateur connaît mieux que l'algorithme le domaine dans lequel chercher.

---

## Détail des étapes techniques

### Étape 1 — Sélection des prédicteurs (LLM)

Le LLM reçoit la liste complète des 79 prédicteurs Margolis et le descriptif du projet, et identifie ceux que le projet est susceptible d'impacter. Pour chaque prédicteur retenu, il fournit un score de pertinence (1 à 5) et une justification en une phrase.

Un matching par mots-clés serait trop rigide ici : "parentalité" ne matche pas "Social integration", "séances de sport" ne matche pas "Physical activity". Le LLM comprend les liens indirects — un atelier cuisine peut impacter "Autonomy" et "Self-efficacy" sans que ces mots apparaissent dans la description du projet.

---

### Étape 2a — Génération des requêtes (déterministe)

Pour chaque prédicteur, jusqu'à **10 requêtes** sont construites mécaniquement par combinaison des termes disponibles. Aucun token LLM consommé — reproductible à l'identique pour les mêmes inputs.

Exemple complet pour le prédicteur `family support`, projet parentalité, `search_context = "parenting support"` :

| # | Groupe | Requête générée |
|---|---|---|
| Q1 | 🔬 meta | `family support meta-analysis` |
| Q2 | 🔬 meta | `family support children aged meta-analysis` |
| Q3 | 🔬 meta | `family support childhood parenting meta-analysis` |
| Q4 | 🔬 meta | `family support childhood systematic review` |
| Q5 | 🔬 meta | `family support wellbeing systematic review` |
| Q6 | 📖 ctx | `family support parenting wellbeing` |
| Q7 | 📖 ctx | `family support parenting intervention` |
| Q8 | 📖 ctx | `family support parenting` |
| Q9 | 📖 ctx | `family support parenting children aged` |
| Q10 | 📖 ctx | `family support parenting children aged` |

Les requêtes 🔬 ciblent les méta-analyses et revues systématiques, qui contiennent le plus souvent des effect sizes explicites dans l'abstract. Les requêtes 📖 élargissent le pool vers les études primaires pour les prédicteurs peu couverts par des méta-analyses, en profitant du `search_context` pour rester dans le bon domaine.

---

### Étape 2b — Recherche OpenAlex

Chaque requête est soumise à l'API OpenAlex. Pour chaque requête, on récupère `OPENALEX_FETCH` résultats triés par `relevance_score` natif (BM25/full-text search), puis on conserve le **top `TOP_PER_QUERY`** (5 par défaut). Les doublons par DOI sont supprimés entre requêtes et entre prédicteurs.

Le pool résultant contient au maximum `nb_requêtes × TOP_PER_QUERY` articles uniques, soit ~50 candidats par prédicteur dans la configuration par défaut.

**Pourquoi OpenAlex ?** API gratuite, sans clé, 250M+ publications. Fournit les abstracts dans un format index inversé `{mot: [positions]}` (contrainte de droits d'auteur) que le moteur reconstitue automatiquement. Couverture ~80% du corpus académique mondial.

---

### Étape 2c — Sélection LLM

Le LLM reçoit en une seule requête les titres + débuts d'abstract des ~50 candidats du pool et retourne une liste d'indices JSON correspondant aux `ARTICLES_PER_PREDICTOR` meilleurs articles. Il raisonne sur trois critères dans cet ordre de priorité :

1. **Compatibilité projet** — la population et le contexte de l'article correspondent-ils au groupe cible et au domaine du projet ? C'est le filtre principal : un article sur les patients en soins palliatifs est exclu d'un projet parentalité, même s'il mesure un prédicteur pertinent.
2. **Lien avec le prédicteur** — l'article mesure-t-il directement le prédicteur ou un concept très proche ?
3. **Présence probable d'un effect size** — les méta-analyses, RCTs et revues systématiques sont favorisées.

Ce filtre consomme très peu de tokens (`max_tokens=60` — juste une liste d'indices). En cas d'échec (réponse mal formée), le fallback est un tri par citations décroissantes.

---

### Étape 3 — Extraction LLM

Le LLM lit l'abstract de chaque article retenu et extrait les champs nécessaires à Boussole. Les effect sizes dans les abstracts sont exprimés de façons très hétérogènes (`d = 0.41`, `Hedges' g of 0.3`, `SMD = 0.52`) — le LLM comprend le contexte et sélectionne la valeur la plus directement liée au prédicteur ciblé.

Il retourne également :
- `source_text` — le passage exact de l'abstract d'où vient la valeur, permettant à l'évaluateur de vérifier l'extraction
- `selection_reason` — une phrase expliquant pourquoi l'article a été jugé pertinent pour ce prédicteur dans ce projet
- `confidence` — un score 1-5 indiquant la certitude de l'extraction (un score ≤ 3 doit toujours être vérifié manuellement)

---

## Output

### Colonnes du DataFrame de résultats

| Colonne | Description |
|---|---|
| `predictor` | Nom du prédicteur Margolis (EN) |
| `domain` | Domaine Margolis (FR) |
| `pred_relevance` | Score de pertinence LLM (1-5) |
| `pred_justif` | Justification de la sélection du prédicteur |
| `search_queries` | Requêtes envoyées à OpenAlex (séparées par ` \| `) |
| `title` | Titre de l'article |
| `authors` | Auteurs (2 premiers) |
| `year` | Année de publication |
| `doi` | DOI de l'article |
| `cited_by` | Nombre de citations |
| `open_access` | Article en open access ? |
| `effect_size` | Taille d'effet extraite (float) |
| `effect_type` | Type : Cohen's d / Hedge's g / SMD |
| `effect_direction` | Direction : positif / négatif |
| `effect_duration` | Durée de persistance de l'effet |
| `art_relevance` | Pertinence de l'article (1-5) |
| `confidence` | Confiance de l'extraction (1-5) |
| `source_text` | Passage exact de l'abstract d'où vient l'effect size |
| `selection_reason` | Justification LLM du choix de l'article |
| `note` | Note complémentaire du LLM |

### Export Excel — 4 onglets

| Onglet | Contenu |
|---|---|
| `Boussole — avec effect size` | Format exact de l'onglet *Notes export*, toutes les lignes avec une valeur extraite — l'évaluateur choisit quelle ligne retenir |
| `Boussole — sans effect size` | Même format, articles sans effect size trouvé — à compléter manuellement |
| `Tous les articles` | Toutes les colonnes techniques |
| `Prédicteurs` | Prédicteurs sélectionnés avec scores et justifications |

L'onglet **Boussole** respecte exactement les colonnes du fichier *Notes export* fourni par Elements Impact : `Project`, `Project description`, `Tags`, `Group`, `Domain`, `Predictor`, `Effect size`, `Effect duration`, `Note (raw)`, `Article links`.

---

## Limites connues

- **Abstracts uniquement** — les effect sizes reportés uniquement dans les tableaux ou la section "Results" ne sont pas capturés. Environ 20% des articles dans OpenAlex n'ont pas d'abstract disponible.
- **Couverture OpenAlex** — ~80% du corpus académique. Certains articles spécialisés ou très récents peuvent être absents.
- **Inputs en anglais recommandés** — la génération des requêtes est purement lexicale (tokenisation). Des inputs en français produisent des tokens peu utiles pour des requêtes académiques en anglais.
- **Extraction probabiliste** — le score `confidence` (1-5) permet d'identifier les extractions incertaines. Un score ≤ 3 doit toujours être vérifié manuellement via le `source_text`.
- **Le moteur propose, l'évaluateur valide** — l'onglet "Boussole — avec effect size" liste toutes les options trouvées. C'est à l'évaluateur de choisir la valeur la plus pertinente et de vérifier le passage source.

---

## Perspectives d'amélioration

- **Texte complet** — parsing PDF des articles open-access pour accéder aux tableaux de résultats, pas seulement aux abstracts
- **Cache inter-projets** — éviter de re-fetcher les mêmes articles pour des prédicteurs déjà explorés sur d'autres projets
- **Fine-tuning** d'un extracteur NLP spécialisé sur le corpus Boussole pour l'étape 3, afin de réduire la dépendance au LLM généraliste
- **Scoring sémantique** — remplacer le tri BM25 d'OpenAlex par un re-ranking par embeddings pour mieux capturer la pertinence sémantique entre l'article et le prédicteur
