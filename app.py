import gradio as gr
import plotly.graph_objects as go
import numpy as np
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---- Load model ----
pipeline = joblib.load("arxiv_classifier_pipeline.joblib")
le_mb    = joblib.load("label_encoder.joblib")
with open("custom_phrase_stop.pkl", "rb") as f:
    CUSTOM_PHRASE_STOP = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---- Domain metadata ----
DOMAIN_META = {
    'cs.AI_NE': {'label': 'AI & Neural Computing',        'color': '#FF6B9D'},
    'cs.CE':    {'label': 'Computational Engineering',     'color': '#B060FF'},
    'cs.CV':    {'label': 'Computer Vision',               'color': '#60CFFF'},
    'cs.DS':    {'label': 'Data Structures & Algorithms',  'color': '#60FFB0'},
    'cs.IT':    {'label': 'Information Theory',            'color': '#FFB060'},
    'cs.PL':    {'label': 'Programming Languages',         'color': '#FF60CF'},
    'cs.SY':    {'label': 'Systems & Control',             'color': '#6090FF'},
    'math.AC':  {'label': 'Abstract Algebra',              'color': '#FFD700'},
    'math.GR':  {'label': 'Group Theory',                  'color': '#FF8C60'},
    'math.ST':  {'label': 'Statistics & Probability',      'color': '#96CEB4'},
}

# 3D semantic positions (from EDA Jaccard similarity structure)
DOMAIN_3D_POS = {
    'cs.AI_NE': ( 1.0,  1.0,  0.0),
    'cs.CE':    ( 0.8,  0.5,  0.2),
    'cs.CV':    ( 1.2,  0.8,  0.1),
    'cs.DS':    ( 0.5,  1.2,  0.3),
    'cs.IT':    ( 0.3,  0.8,  0.5),
    'cs.PL':    (-0.5,  1.0,  0.2),
    'cs.SY':    ( 0.6,  0.3,  0.8),
    'math.AC':  (-1.5, -0.5,  0.5),
    'math.GR':  (-1.2, -1.0,  0.3),
    'math.ST':  (-0.3, -1.2,  0.8),
}

# ---- Preprocessing ----
def preprocess(text):
    text = str(text).lower()
    for phrase in CUSTOM_PHRASE_STOP:
        text = text.replace(phrase, '')
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\[a-z]+\{.*?\}', '', text)
    text = re.sub(r'\\[a-z]+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def extract_keywords(text, n=10):
    cleaned = preprocess(text)
    extra_stop = {'also', 'using', 'show', 'paper', 'propose', 'based',
                  'approach', 'method', 'result', 'study', 'work', 'present',
                  'use', 'two', 'one', 'new', 'model', 'different', 'used'}
    words = [w for w in cleaned.split() if w not in extra_stop and len(w) > 3]
    return [w for w, _ in Counter(words).most_common(n)]

def build_3d_clusters(top_k_classes, top_k_scores):
    traces = []
    np.random.seed(42)
    for cls, score in zip(top_k_classes, top_k_scores):
        cx, cy, cz = DOMAIN_3D_POS.get(cls, (0, 0, 0))
        color  = DOMAIN_META.get(cls, {}).get('color', '#FFFFFF')
        label  = DOMAIN_META.get(cls, {}).get('label', cls)
        n      = max(40, int(180 * score))
        spread = max(0.08, 0.35 * (1 - score))
        x = np.random.normal(cx, spread, n)
        y = np.random.normal(cy, spread, n)
        z = np.random.normal(cz, spread, n)
        dist  = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
        sizes = np.clip(7 - dist * 12, 2, 7)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=sizes, color=color, opacity=0.75, line=dict(width=0)),
            name=f'{label} ({score*100:.1f}%)',
            hovertemplate=f'<b>{label}</b><br>Confidence: {score:.1%}<extra></extra>'
        ))
    return traces

# ---- Main predict function ----
def predict(abstract, top_k):
    if not abstract or len(abstract.strip()) < 20:
        err = ("<div style='background:#1A1535;border-radius:16px;padding:40px;"
               "text-align:center;color:#FF6B9D;font-family:Inter,sans-serif;'>"
               "⚠️ Please enter a research abstract of at least 20 characters.</div>")
        return err, None, err

    top_k   = int(top_k)
    cleaned = preprocess(abstract)
    proba   = pipeline.predict_proba([cleaned])[0]
    classes = le_mb.classes_

    sorted_idx    = np.argsort(proba)[::-1]
    top_k_idx     = sorted_idx[:top_k]
    top_k_classes = classes[top_k_idx]
    top_k_scores  = proba[top_k_idx]

    top1_cls   = top_k_classes[0]
    top1_score = top_k_scores[0]
    top1_label = DOMAIN_META.get(top1_cls, {}).get('label', top1_cls)
    top1_color = DOMAIN_META.get(top1_cls, {}).get('color', '#FF6B9D')

    if top1_score >= 0.6:
        conf_text  = "✅ High Confidence — Auto-routed"
        conf_color = "#60FFB0"
        insight    = (f"The abstract is most similar to papers in {top1_label}. "
                      f"The model is highly confident based on strong domain-specific "
                      f"vocabulary alignment detected by TF-IDF feature weighting.")
    elif top1_score >= 0.4:
        conf_text  = "⚡ Moderate Confidence"
        conf_color = "#FFB060"
        insight    = (f"The abstract shows characteristics of {top1_label} with some overlap "
                      f"with adjacent domains. Review the top-{top_k} predictions for full context.")
    else:
        conf_text  = "⚠️ Low Confidence — Human Review Recommended"
        conf_color = "#FF6B9D"
        insight    = (f"This abstract spans multiple research domains. The top prediction is "
                      f"{top1_label}, but vocabulary overlap with adjacent fields makes "
                      f"classification uncertain. Manual review is recommended.")

    keywords = extract_keywords(abstract)

    # ---- Top domain card ----
    top_card = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
    </style>
    <div style='background:linear-gradient(135deg,{top1_color}22,#0F0C2A);
                border:1px solid {top1_color}44; border-radius:16px;
                padding:28px; font-family:Inter,sans-serif; position:relative;
                overflow:hidden; min-height:240px;'>

        <div style='display:inline-flex;align-items:center;gap:8px;
                    background:rgba(255,255,255,0.08);border-radius:20px;
                    padding:6px 14px;margin-bottom:18px;'>
            <span>🏆</span>
            <span style='font-size:0.78rem;color:#E0E0FF;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.08em;'>
                Top Predicted Domain
            </span>
        </div>

        <div style='font-family:Space Grotesk,sans-serif;font-size:2rem;
                    font-weight:700;color:{top1_color};margin-bottom:14px;
                    line-height:1.2;'>
            {top1_label}
        </div>

        <div style='font-size:0.75rem;color:#8899BB;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;'>Confidence</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:2.8rem;
                    font-weight:700;color:{top1_color};line-height:1;'>
            {top1_score*100:.2f}%
        </div>

        <div style='margin-top:14px;background:rgba(255,255,255,0.1);
                    border-radius:4px;height:6px;'>
            <div style='background:linear-gradient(90deg,{top1_color},{top1_color}66);
                        width:{top1_score*100:.1f}%;height:100%;border-radius:4px;
                        transition:width 0.6s ease;'></div>
        </div>

        <div style='margin-top:14px;display:inline-block;
                    background:rgba(255,255,255,0.06);border-radius:20px;
                    padding:5px 14px;font-size:0.78rem;color:{conf_color};
                    border:1px solid {conf_color}44;'>
            {conf_text}
        </div>

        <div style='position:absolute;top:18px;right:18px;
                    background:rgba(255,255,255,0.06);border-radius:8px;
                    padding:5px 10px;font-family:monospace;font-size:0.78rem;
                    color:#8899BB;'>
            {top1_cls}
        </div>
    </div>
    """

    # ---- Legend rows ----
    legend_rows = ""
    for cls, score in zip(top_k_classes, top_k_scores):
        c = DOMAIN_META.get(cls, {}).get('color', '#FFF')
        l = DOMAIN_META.get(cls, {}).get('label', cls)
        legend_rows += f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
            <div style='width:9px;height:9px;border-radius:50%;background:{c};
                        flex-shrink:0;box-shadow:0 0 6px {c};'></div>
            <div style='flex:1;font-size:0.85rem;color:#E0E0FF;'>{l}</div>
            <div style='font-weight:600;color:{c};font-size:0.88rem;'>{score*100:.2f}%</div>
        </div>"""

    # ---- Keyword pills ----
    pills = "".join(
        f"<span style='display:inline-block;background:rgba(176,96,255,0.12);"
        f"border:1px solid rgba(176,96,255,0.35);color:#D0B0FF;padding:4px 12px;"
        f"border-radius:20px;font-size:0.78rem;margin:4px;'>{kw}</span>"
        for kw in keywords
    )

    # ---- Bottom HTML ----
    bottom = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
    </style>

    <div style='background:#1A1535;border-radius:16px;padding:22px;
                margin-bottom:14px;font-family:Inter,sans-serif;
                border:1px solid rgba(255,255,255,0.06);'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
            <span>📦</span>
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.95rem;
                        font-weight:600;color:#F0F4FF;'>Domain Distribution (Top {top_k})</div>
        </div>
        <div style='font-size:0.78rem;color:#8899BB;margin-bottom:14px;'>
            Each cluster represents a research domain. Closer clusters share more vocabulary.
        </div>
        <div style='border-top:1px solid rgba(255,255,255,0.07);padding-top:14px;'>
            {legend_rows}
        </div>
    </div>

    <div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;
                font-family:Inter,sans-serif;'>
        <div style='background:#1A1535;border-radius:16px;padding:22px;
                    border:1px solid rgba(255,255,255,0.06);'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>
                <span>💡</span>
                <div style='font-family:Space Grotesk,sans-serif;font-size:0.95rem;
                            font-weight:600;color:#F0F4FF;'>Prediction Insight</div>
            </div>
            <div style='position:relative;padding-left:14px;'>
                <div style='position:absolute;left:0;top:0;bottom:0;width:3px;
                            background:linear-gradient(180deg,#B060FF,#FF6B9D);
                            border-radius:2px;'></div>
                <p style='color:#C0C8E0;font-size:0.85rem;line-height:1.65;margin:0;'>
                    {insight}
                </p>
            </div>
        </div>
        <div style='background:#1A1535;border-radius:16px;padding:22px;
                    border:1px solid rgba(255,255,255,0.06);'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>
                <span>🏷️</span>
                <div style='font-family:Space Grotesk,sans-serif;font-size:0.95rem;
                            font-weight:600;color:#F0F4FF;'>Extracted Keywords</div>
            </div>
            <div style='line-height:2.2;'>{pills}</div>
        </div>
    </div>
    """

    # ---- 3D plot ----
    traces = build_3d_clusters(top_k_classes, top_k_scores)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text='Semantic Dimension 1',
                                   font=dict(color='#8899BB', size=9)),
                       gridcolor='#2A1F4A', backgroundcolor='#0D0B1E',
                       showbackground=True, showticklabels=False, zeroline=False),
            yaxis=dict(title=dict(text='Semantic Dimension 2',
                                   font=dict(color='#8899BB', size=9)),
                       gridcolor='#2A1F4A', backgroundcolor='#0D0B1E',
                       showbackground=True, showticklabels=False, zeroline=False),
            zaxis=dict(title=dict(text='Semantic Dimension 3',
                                   font=dict(color='#8899BB', size=9)),
                       gridcolor='#2A1F4A', backgroundcolor='#0D0B1E',
                       showbackground=True, showticklabels=False, zeroline=False),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
            bgcolor='#0D0B1E',
            aspectmode='cube',
        ),
        paper_bgcolor='#0D0B1E',
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        legend=dict(font=dict(color='#C0C8E0', size=9),
                    bgcolor='rgba(26,21,53,0.85)',
                    bordercolor='rgba(255,255,255,0.08)',
                    borderwidth=1, x=1.0, y=0.95),
        showlegend=True,
    )

    return top_card, fig, bottom


# ==========================================
# HEADER with rotating radar animation
# ==========================================

HEADER_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

@keyframes radarSweep {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes blink {
    0%,100% { opacity:0.2; }
    50%      { opacity:1; }
}

.r-wrap {
    display:flex; align-items:center; gap:36px;
    padding:40px 5vw;
    width:100%;
    box-sizing:border-box;
    background:linear-gradient(135deg,#100D2A 0%,#180E3A 50%,#0D0B1E 100%);
    border-radius:0;
    margin:0 0 20px 0;
    border-bottom:1px solid rgba(176,96,255,0.18);
    position:relative; overflow:hidden;
}

.r-icon {
    position:relative; width:220px; height:220px; flex-shrink:0;
}
.r-ring {
    position:absolute; border-radius:50%;
    border:2px solid rgba(255,20,147,0.35);
    top:50%; left:50%; transform:translate(-50%,-50%);
}
.r-ring-1 { width:100%; height:100%; border-color:rgba(255,20,147,0.35); }
.r-ring-2 { width:68%; height:68%; border-color:rgba(255,20,147,0.55); }
.r-ring-3 { width:36%; height:36%; border-color:rgba(255,20,147,0.85); }

.r-sweep {
    position:absolute; width:50%; height:3px;
    top:50%; left:50%; transform-origin:left center;
    animation:radarSweep 3s linear infinite;
    background:linear-gradient(90deg,rgba(255,255,255,0.95),transparent);
    border-radius:3px;
}
.r-sweep::after {
    content:''; position:absolute;
    right:-5px; top:-4.5px; width:11px; height:11px;
    background:#FFFFFF; border-radius:50%;
    box-shadow:0 0 16px #FFFFFF, 0 0 30px rgba(255,255,255,0.6);
}

.r-dot {
    position:absolute; border-radius:50%;
    animation:blink 2s ease-in-out infinite;
}
.r-dot-1 { width:7px;height:7px;background:#FF6B9D;
            box-shadow:0 0 10px #FF6B9D;
            top:18%;right:22%;animation-delay:0s; }
.r-dot-2 { width:6px;height:6px;background:#60CFFF;
            box-shadow:0 0 8px #60CFFF;
            bottom:28%;left:18%;animation-delay:0.9s; }
.r-dot-3 { width:5px;height:5px;background:#60FFB0;
            box-shadow:0 0 6px #60FFB0;
            top:58%;right:14%;animation-delay:1.6s; }

.r-title { font-family:'Space Grotesk',sans-serif; font-size:3.4rem;
           font-weight:700; color:#F0F4FF; letter-spacing:0.12em;
           margin:0; line-height:1; }
.r-sub   { font-family:'Inter',sans-serif; font-size:0.95rem;
           color:#8899BB; margin-top:10px; letter-spacing:0.04em; }
.r-stats { display:flex; gap:36px; margin-top:20px; }
.r-sv    { font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
           font-weight:700;
           background:linear-gradient(135deg,#B060FF,#FF6B9D);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           background-clip:text; }
.r-sl    { font-size:0.68rem; color:#8899BB; text-transform:uppercase;
           letter-spacing:0.1em; }

.r-deco  { position:absolute; border-radius:50%;
           border:1px solid rgba(176,96,255,0.07);
           pointer-events:none; }
</style>

<div class='r-wrap'>
    <div class='r-deco' style='width:420px;height:420px;right:-80px;top:-100px;'></div>
    <div class='r-deco' style='width:280px;height:280px;right:-40px;top:-50px;
                                border-color:rgba(255,107,157,0.05);'></div>

    <div class='r-icon'>
        <div class='r-ring r-ring-1'></div>
        <div class='r-ring r-ring-2'></div>
        <div class='r-ring r-ring-3'></div>
        <div class='r-sweep'></div>
        <div class='r-dot r-dot-1'></div>
        <div class='r-dot r-dot-2'></div>
        <div class='r-dot r-dot-3'></div>
    </div>

    <div>
        <p class='r-title'>RADAR</p>
        <p class='r-sub'>Research Abstract Domain Assignment &amp; Ranking</p>
        <div class='r-stats'>
            <div><div class='r-sv'>10</div><div class='r-sl'>Domains</div></div>
            <div><div class='r-sv'>4,844</div><div class='r-sl'>Trained On</div></div>
            <div><div class='r-sv'>81.65%</div><div class='r-sl'>Macro F1</div></div>
            <div><div class='r-sv'>96.0%</div><div class='r-sl'>Top-3 Coverage</div></div>
        </div>
    </div>
</div>
"""

FOOTER_HTML = """
<div style='text-align:center;padding:24px 20px;margin-top:12px;
            border-top:1px solid rgba(255,255,255,0.06);
            font-family:Inter,sans-serif;font-size:0.8rem;color:#8899BB;
            display:flex;align-items:center;justify-content:center;gap:20px;
            flex-wrap:wrap;'>
    <span>❤️ Built by <strong style="color:#B060FF;">Rishika</strong></span>
    <span style='color:rgba(255,255,255,0.15);'>|</span>
    <span>📄 Data Source: arXiv Abstracts</span>
    <span style='color:rgba(255,255,255,0.15);'>|</span>
    <span>🔬 DRDO Internship Project</span>
</div>
"""

CUSTOM_CSS = """
body, .gradio-container {
    background: #0D0B1E !important;
    color: #F0F4FF !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 0 20px 0 !important;
}
.gradio-container > .main, .gradio-container .contain {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 20px !important;
}
label > span {
    color: #8899BB !important;
    font-family: Space Grotesk, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
textarea {
    background: #100D2A !important;
    border: 1px solid rgba(176,96,255,0.25) !important;
    border-radius: 12px !important;
    color: #E0E0FF !important;
    font-family: Inter, sans-serif !important;
    font-size: 0.9rem !important;
    padding: 14px !important;
    line-height: 1.6 !important;
}
textarea:focus {
    border-color: #B060FF !important;
    box-shadow: 0 0 0 2px rgba(176,96,255,0.2) !important;
    outline: none !important;
}
input[type=range] { accent-color: #B060FF !important; }
button.primary {
    background: linear-gradient(135deg, #B060FF 0%, #FF6B9D 100%) !important;
    color: white !important;
    font-family: Space Grotesk, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    box-shadow: 0 4px 20px rgba(176,96,255,0.3) !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(176,96,255,0.45) !important;
}
.gr-block, .gr-box, .gr-panel { background: transparent !important; border: none !important; }
.js-plotly-plot { border-radius: 16px !important; overflow: hidden !important; }
"""

# ---- Two detailed examples only, enough to test the app end-to-end ----
EXAMPLES = [
    ["We propose a neuroevolutionary framework for training convolutional neural "
     "networks on large-scale image recognition tasks. Rather than relying on manually "
     "designed architectures, our method uses a genetic algorithm to evolve both the "
     "network topology and hyperparameters, guided by a fitness function based on "
     "validation accuracy and computational cost. We evaluate the evolved architectures "
     "on standard visual recognition benchmarks including CIFAR-10 and ImageNet, and "
     "compare their performance and inference latency against hand-designed convolutional "
     "networks. Our results show that evolved architectures can match or exceed the "
     "accuracy of manually designed networks while requiring significantly fewer parameters, "
     "suggesting that evolutionary search is a viable alternative to manual architecture "
     "design for vision tasks.", 3],

    ["We study the problem of sparse signal recovery from noisy linear measurements in "
     "the presence of Gaussian noise, a setting that arises naturally in compressed "
     "sensing and high-dimensional statistics. We derive minimax lower bounds on the "
     "estimation error under a sparsity constraint on the underlying signal, and propose "
     "an estimator based on convex relaxation techniques that achieves the optimal "
     "convergence rate up to logarithmic factors in the ambient dimension. Our analysis "
     "draws a direct connection between information-theoretic channel capacity bounds "
     "and classical statistical estimation theory, showing that the two perspectives "
     "yield matching rates under mild regularity conditions. We further illustrate the "
     "practical performance of the proposed estimator through simulations across a range "
     "of signal-to-noise ratios and sparsity levels.", 3],
]

# ==========================================
# BUILD GRADIO APP
# ==========================================

with gr.Blocks(title="RADAR — Research Abstract Domain Auto-Routing") as demo:

    gr.HTML(HEADER_HTML)

    gr.HTML("""
    <div style='font-family:Space Grotesk,sans-serif;font-size:0.72rem;
                color:#8899BB;text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:8px;'>📄 Your Research Abstract</div>
    """)
    abstract_input = gr.Textbox(
        placeholder="Paste a research abstract here...",
        lines=7, max_lines=16, show_label=False
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gr.HTML("""
            <div style='font-family:Space Grotesk,sans-serif;font-size:0.72rem;
                        color:#8899BB;text-transform:uppercase;letter-spacing:0.12em;
                        margin-bottom:4px;'>Select Top K Predictions</div>
            <div style='font-size:0.75rem;color:#8899BB;font-family:Inter,sans-serif;
                        margin-bottom:6px;'>Choose how many top domains to display</div>
            """)
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                show_label=False
            )
        with gr.Column(scale=1, min_width=180):
            gr.HTML("<div style='height:30px'></div>")
            classify_btn = gr.Button(
                "✦  Predict Domain",
                variant="primary", size="lg"
            )

    gr.HTML("""
    <div style='font-family:Space Grotesk,sans-serif;font-size:0.7rem;
                color:#8899BB;text-transform:uppercase;letter-spacing:0.12em;
                margin:14px 0 8px;'>Try an example</div>
    """)
    gr.Examples(
        examples=EXAMPLES,
        inputs=[abstract_input, top_k_slider],
        label=""
    )

    top_domain_output = gr.HTML()

    gr.HTML("""
    <div style='font-family:Space Grotesk,sans-serif;font-size:0.72rem;color:#8899BB;
                text-transform:uppercase;letter-spacing:0.12em;margin:16px 0 6px;'>
        📦 Domain Distribution — 3D Semantic View
    </div>
    """)
    plot_output = gr.Plot(label="")

    bottom_output = gr.HTML()

    gr.HTML(FOOTER_HTML)

    classify_btn.click(
        fn=predict,
        inputs=[abstract_input, top_k_slider],
        outputs=[top_domain_output, plot_output, bottom_output]
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
