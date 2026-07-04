# RADAR — Research Abstract Domain Assignment & Ranking

RADAR is a Gradio app that classifies a research paper abstract into one of 10
arXiv-style domains (e.g. Computer Vision, Information Theory, Group Theory)
using a trained TF-IDF + classifier pipeline, and visualizes the prediction
confidence as an interactive 3D semantic cluster plot.

**Model stats:** 10 domains · trained on 4,844 abstracts · 81.65% macro F1 ·
96.0% top-3 coverage.

## Project structure

```
.
├── app.py                          # Gradio app
├── requirements.txt                # Python dependencies
├── arxiv_classifier_pipeline.joblib   # trained sklearn pipeline (add this)
├── label_encoder.joblib               # label encoder (add this)
├── custom_phrase_stop.pkl             # custom stopword phrases (add this)
└── README.md
```

> ⚠️ The three model artifact files above are **not included** in this repo
> (they're binary model files). Copy them into the project root before
> running or deploying — see below.

## Run locally

```bash
git clone https://github.com/<your-username>/radar-app.git
cd radar-app

# add your model files here:
#   arxiv_classifier_pipeline.joblib
#   label_encoder.joblib
#   custom_phrase_stop.pkl

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

The app will start locally at `http://127.0.0.1:7860`.

## Deploy

### Option A — Hugging Face Spaces (recommended, free, built for Gradio)

1. Create a free account at https://huggingface.co if you don't have one.
2. Go to **New Space** → choose **Gradio** as the SDK → name it (e.g. `radar`).
3. Either:
   - Clone the Space repo it gives you and push this project's files into it
     (including the 3 model files), or
   - Upload the files directly through the Space's "Files" web UI.
4. Hugging Face installs `requirements.txt` and runs `app.py` automatically.
   Your app will be live at `https://huggingface.co/spaces/<you>/radar`.

```bash
git clone https://huggingface.co/spaces/<your-username>/radar
cd radar
cp /path/to/radar-app/app.py .
cp /path/to/radar-app/requirements.txt .
cp /path/to/*.joblib /path/to/*.pkl .
git add .
git commit -m "Deploy RADAR"
git push
```

### Option B — Render / Railway / any VM

Any host that can run `pip install -r requirements.txt && python app.py`
works. Just make sure the model files ship with the deployment and the port
Gradio binds to is exposed (Gradio respects the `PORT` env var on most PaaS
hosts, or you can pass `server_port=int(os.environ.get("PORT", 7860))` to
`demo.launch()`).

## Notes

- Model files are excluded from git via size — if they're small enough for a
  normal GitHub repo (under ~100MB), you can commit them directly. Otherwise
  use [Git LFS](https://git-lfs.com/) or host them on Hugging Face Hub /
  Google Drive and download them at app startup.

## Credit

Built by Rishika — DRDO Internship Project. Data source: arXiv abstracts.
