# RADAR — Research Abstract Domain Assignment & Ranking

🔗 **Live demo:** https://huggingface.co/spaces/rishikaml2026/radar

## Problem Statement

DRDO operates through multiple specialized labs (DRDL, CVRDE, CAIR, DMRL, NPOL,
DIPAS, etc.), each generating a continuous stream of technical reports, project
abstracts, internal research notes, and literature survey summaries. In
practice, these documents aren't consistently tagged by research domain when
filed, which means researchers across labs often can't quickly discover
relevant prior work, technical notes get archived in the wrong repository
bucket, and duplicated research effort happens simply because nobody could
find that a related study already existed in another lab. This is a
well-documented real pain point in large multi-lab R&D organizations, not a
hypothetical one.

## Proposed Solution

Build an NLP-based classifier that reads the free-text title and
abstract/summary of a technical document and automatically assigns it to the
correct research domain category (e.g., Electronics & Communication Systems,
Materials Science, Aeronautics & Propulsion, AI/Software Systems, Missile &
Strategic Systems, etc.), enabling consistent auto-tagging for a searchable
cross-lab knowledge repository instead of relying on manual, inconsistent
filing.

The key design choice that makes this more than a basic classifier: many real
documents genuinely span more than one domain (a paper on a new composite
material for missile nose cones touches both Materials Science and Missile
Systems). So instead of forcing a single hard label, the system supports
top-2/top-3 ranked predictions with confidence scores, and flags genuinely
ambiguous documents for human review rather than silently mislabeling them.
This mirrors a real organizational need — cross-domain tagging — rather than
a toy single-label exercise.

## This prototype

DRDO's internal technical reports aren't public, so this prototype is built
and validated on **arXiv abstracts** across 10 research domains as an
accessible stand-in dataset with the same core property that matters here:
genuine cross-domain overlap between related fields (e.g. AI & Neural
Computing vs. Computer Vision, or Abstract Algebra vs. Group Theory). The
same pipeline — TF-IDF + classifier, top-k ranked output, confidence-based
human-review flagging — is directly transferable to DRDO's internal document
categories once trained on that internal corpus.

**Model stats:** 10 domains · trained on 4,844 abstracts · 81.65% macro F1 ·
96.0% top-3 coverage.

## Project structure

```
.
├── app.py                              # Gradio app (inference + UI)
├── requirements.txt                    # Python dependencies
├── arxiv_classifier_pipeline.joblib    # trained sklearn pipeline
├── label_encoder.joblib                # label encoder
├── custom_phrase_stop.pkl              # custom stopword phrases
└── notebooks/
    ├── 01_eda.ipynb                    # exploratory data analysis
    ├── 02_model_training.ipynb         # preprocessing + model training
    └── 03_gradio_app.ipynb             # original Colab app build/deploy
```
