"""Look and feel of the Streamlit app: CSS plus small HTML builders (no model logic here)."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --ink:#0b1412; --muted:#5d6e69; --line:#e3ebe8; --bg:#f3f6f5; --card:#ffffff;
  --dark:#07110f; --dark2:#0c2420; --teal:#17c3a2; --teal-d:#0f7f6b; --green:#1f6a5c;
  --mint:#d8f5ec; --red:#ef4a4a; --amber:#f4a62a; --ok:#19b37d;
}
html, body, [class*="css"], .stApp { font-family:'Inter','Segoe UI',system-ui,sans-serif !important; color:var(--ink); }
.stApp { background:var(--bg); }
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stDecoration"] { display:none !important; }
.block-container { max-width:1280px; padding:1.1rem 1.4rem 3rem !important; }
h1,h2,h3,h4 { letter-spacing:-0.02em; }

/* ---------- hero ---------- */
.hero{ position:relative; overflow:hidden; border-radius:34px; padding:22px 38px 40px; color:#fff;
  background: radial-gradient(900px 420px at 78% 108%, rgba(23,195,162,.55), transparent 62%),
              radial-gradient(700px 360px at 8% -10%, rgba(31,106,92,.55), transparent 60%),
              linear-gradient(160deg,#050c0b 0%,#0a1c19 55%,#07110f 100%); }
.hero:before{ content:""; position:absolute; inset:0; opacity:.35;
  background-image: radial-gradient(rgba(255,255,255,.18) 1px, transparent 1px); background-size:26px 26px;
  mask-image: linear-gradient(180deg, transparent, #000 40%, transparent); }
.nav{ position:relative; display:flex; align-items:center; justify-content:space-between; gap:16px; z-index:2; }
.brand{ font-weight:800; font-size:1.05rem; letter-spacing:-.01em; display:flex; align-items:center; gap:10px; }
.brand i{ width:30px; height:30px; border-radius:10px; display:inline-grid; place-items:center; font-style:normal;
  background:linear-gradient(135deg,var(--teal),#0c6b5a); box-shadow:0 6px 18px rgba(23,195,162,.45); }
.navlinks{ display:flex; gap:26px; font-size:.86rem; color:rgba(255,255,255,.72); }
.pill{ border-radius:999px; padding:8px 16px; font-size:.8rem; font-weight:600; border:1px solid rgba(255,255,255,.22);
  background:rgba(255,255,255,.06); color:#fff; }
.pill.solid{ background:#fff; color:#07110f; border-color:#fff; }
.hero-grid{ position:relative; z-index:2; display:grid; grid-template-columns:1.25fr 1fr; gap:30px; align-items:center; margin-top:34px; }
.chip{ display:inline-flex; align-items:center; gap:8px; font-size:.78rem; padding:6px 14px; border-radius:999px;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); color:rgba(255,255,255,.85); margin-bottom:18px; }
.chip b{ width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px rgba(23,195,162,.25); animation:pulse 2s infinite; }
@keyframes pulse{ 50%{ box-shadow:0 0 0 9px rgba(23,195,162,0);} }
.hero .h1{ font-size:3.25rem; line-height:1.04; font-weight:800; margin:0 0 16px; color:#fff; }
.hero .h1 span{ background:linear-gradient(90deg,#5ff0cf,#17c3a2 60%,#9bf5e0); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p.lead{ color:rgba(255,255,255,.7); font-size:1.02rem; line-height:1.6; max-width:520px; margin:0 0 24px; }
.cta{ display:flex; gap:12px; flex-wrap:wrap; }
.cta .pill{ padding:12px 24px; font-size:.9rem; }
.glass-stack{ position:relative; height:330px; }
.glass{ position:absolute; border-radius:22px; padding:16px 18px; backdrop-filter:blur(14px);
  background:linear-gradient(145deg, rgba(255,255,255,.14), rgba(255,255,255,.04)); border:1px solid rgba(255,255,255,.2);
  box-shadow:0 24px 50px rgba(0,0,0,.4); animation:floaty 6s ease-in-out infinite; }
.glass small{ color:rgba(255,255,255,.62); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; }
.glass .big{ font-size:1.9rem; font-weight:800; margin-top:2px; }
.g1{ left:0; top:0; width:54%; } .g2{ right:0; top:84px; width:48%; animation-delay:-2s; } .g3{ left:14%; top:214px; width:56%; animation-delay:-4s; }
@keyframes floaty{ 50%{ transform:translateY(-12px);} }
.mini-bars{ display:flex; gap:5px; align-items:flex-end; height:42px; margin-top:8px; }
.mini-bars span{ flex:1; background:linear-gradient(180deg,var(--teal),#0f7f6b); border-radius:5px 5px 2px 2px; opacity:.9; }
.statrow{ position:relative; z-index:2; display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:34px; }
.stat{ border-radius:20px; padding:16px 18px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); transition:.25s; }
.stat:hover{ transform:translateY(-4px); background:rgba(255,255,255,.1); }
.stat .v{ font-size:1.7rem; font-weight:800; } .stat .l{ font-size:.76rem; color:rgba(255,255,255,.62); }

/* ---------- sections ---------- */
.eyebrow{ font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; color:var(--teal-d); font-weight:700; margin:34px 0 8px; display:flex; gap:8px; align-items:center; }
.eyebrow:before{ content:""; width:8px; height:8px; border-radius:50%; background:var(--teal); }
.headline{ font-size:2.35rem; line-height:1.1; font-weight:800; margin:0 0 6px; max-width:760px; }
.headline span{ color:#7b8c87; font-weight:700; }
.sub{ color:var(--muted); max-width:640px; margin-bottom:18px; }

/* streamlit tabs as a pill navigation */
[data-testid="stTabs"] [role="tablist"]{ gap:8px; background:#fff; padding:6px; border-radius:999px; width:fit-content; margin:20px 0 6px;
  border:1px solid var(--line); box-shadow:0 8px 24px rgba(11,20,18,.05); }
[data-testid="stTabs"] > div:first-child{ border-bottom:0 !important; box-shadow:none !important; }
[data-testid="stTab"]{ height:42px; border-radius:999px !important; padding:0 24px; font-weight:600; color:var(--muted); background:transparent;
  border:0 !important; box-shadow:none !important; transition:.2s; }
[data-testid="stTab"] *{ color:inherit !important; font-weight:600 !important; }
[data-testid="stTab"]:hover{ background:var(--mint); }
[data-testid="stTab"][aria-selected="true"]{ background:var(--dark) !important; color:#fff !important; }
[data-testid="stTab"]::after, [data-testid="stTab"]::before, [data-testid="stTabs"] > div:first-child::after{ display:none !important; }
.react-aria-SelectionIndicator{ display:none !important; }

/* cards from bordered containers */
[class*="st-key-card_"]{ background:var(--card) !important; border:1px solid var(--line) !important; border-radius:28px !important;
  box-shadow:0 18px 44px rgba(11,20,18,.06); padding:16px 20px !important; }
.st-key-risk_card{ background:linear-gradient(165deg,#0a1c19,#0d2b25 60%,#0b1f1b) !important; border:0 !important; color:#fff; border-radius:28px !important; }
.st-key-risk_card *{ color:inherit; }
.card-title{ font-size:1.05rem; font-weight:700; margin:2px 0 10px; display:flex; align-items:center; gap:8px; }
.card-title em{ font-style:normal; font-size:.7rem; padding:3px 10px; border-radius:999px; background:var(--mint); color:var(--teal-d); font-weight:700; }
.sig{ display:grid; grid-template-columns:150px 1fr 40px; gap:10px; align-items:center; margin:9px 0; font-size:.82rem; color:rgba(255,255,255,.82); }
.sig .track{ height:9px; border-radius:99px; background:rgba(255,255,255,.12); overflow:hidden; }
.sig .fill{ height:100%; border-radius:99px; animation:grow .9s ease-out; }
@keyframes grow{ from{ width:0 !important; } }
.sig b{ text-align:right; font-weight:700; }
.note{ font-size:.74rem; color:rgba(255,255,255,.55); margin-top:10px; line-height:1.45; }
.verdict{ display:flex; align-items:center; gap:18px; margin-top:6px; }
.verdict .txt .vt{ margin:0; font-size:1.5rem; font-weight:800; } .verdict .txt p{ margin:2px 0 0; font-size:.84rem; color:rgba(255,255,255,.72); }
.badge{ display:inline-block; padding:5px 14px; border-radius:999px; font-size:.78rem; font-weight:800; letter-spacing:.04em; }
.badge.fraud{ background:rgba(239,74,74,.18); color:#ff8c8c; border:1px solid rgba(239,74,74,.5); }
.badge.legit{ background:rgba(25,179,125,.18); color:#6ff0bf; border:1px solid rgba(25,179,125,.5); }
.recent{ display:flex; justify-content:space-between; font-size:.8rem; padding:7px 0; border-bottom:1px dashed rgba(255,255,255,.14); color:rgba(255,255,255,.78); }

/* widgets */
.stButton>button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"]{ border-radius:999px !important; font-weight:700 !important; padding:.55rem 1.4rem !important;
  border:1px solid var(--line) !important; transition:.2s; }
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 10px 22px rgba(11,20,18,.12); border-color:var(--teal) !important; }
[data-testid="stBaseButton-primary"]{ background:linear-gradient(135deg,#17c3a2,#0f7f6b) !important; color:#fff !important; border:0 !important; }
[data-baseweb="select"]>div, .stNumberInput input, .stDateInput input{ border-radius:14px !important; }
[data-testid="stSlider"] [role="slider"]{ background:var(--teal) !important; box-shadow:0 0 0 6px rgba(23,195,162,.2); }
label, [data-testid="stWidgetLabel"] p{ font-weight:600 !important; font-size:.84rem !important; color:#34443f; }

/* tiles + bars + steps */
.tiles{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:6px 0 4px; }
.tile{ background:#fff; border:1px solid var(--line); border-radius:22px; padding:16px 18px; box-shadow:0 12px 30px rgba(11,20,18,.05); transition:.25s; }
.tile:hover{ transform:translateY(-4px); box-shadow:0 20px 40px rgba(11,20,18,.1); }
.tile .l{ font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.tile .v{ font-size:1.85rem; font-weight:800; margin-top:4px; } .tile .v.g{ color:var(--teal-d); } .tile .v.r{ color:var(--red); }
.hbar{ display:grid; grid-template-columns:190px 1fr 56px; gap:12px; align-items:center; margin:8px 0; font-size:.85rem; }
.hbar .track{ height:12px; border-radius:99px; background:#e8efec; overflow:hidden; } .hbar .fill{ height:100%; border-radius:99px; background:linear-gradient(90deg,#0f7f6b,#17c3a2); animation:grow 1s ease-out; }
.hbar.best .fill{ background:linear-gradient(90deg,#f4a62a,#ffd07a); }
.steps{ display:grid; grid-template-columns:1.1fr 1fr 1fr; gap:16px; margin-top:10px; }
.step{ border-radius:26px; padding:26px; min-height:210px; background:#fff; border:1px solid var(--line); transition:.3s; }
.step:hover{ transform:translateY(-6px); box-shadow:0 22px 44px rgba(11,20,18,.1); }
.step.green{ background:linear-gradient(160deg,#27806f,#1a5a4e); color:#fff; border:0; }
.step .n{ width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-weight:800; background:var(--mint); color:var(--teal-d); margin-bottom:14px; }
.step.green .n{ background:rgba(255,255,255,.18); color:#fff; }
.step .st{ margin:0 0 8px; font-size:1.15rem; font-weight:800; } .step p{ margin:0; font-size:.9rem; line-height:1.55; opacity:.82; }
.trust{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px; padding:16px 4px; margin-top:26px; }
.trust span.t{ font-size:.82rem; color:var(--muted); margin-right:8px; }
.trust span.logo{ padding:8px 16px; border-radius:999px; background:#fff; border:1px solid var(--line); font-weight:700; font-size:.82rem; color:#2c3e39; transition:.2s; }
.trust span.logo:hover{ background:var(--dark); color:#fff; transform:translateY(-2px); }
.footer{ margin-top:34px; border-radius:30px; padding:34px 38px; color:#fff; display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:24px;
  background: radial-gradient(520px 240px at 100% 100%, rgba(23,195,162,.35), transparent 60%), linear-gradient(160deg,#050c0b,#0b1f1b); }
.footer .ft{ font-size:1.5rem; margin:0 0 8px; font-weight:800; color:#fff; } .footer p, .footer li{ color:rgba(255,255,255,.66); font-size:.86rem; line-height:1.7; }
.footer ul{ list-style:none; padding:0; margin:0; } .footer b{ font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; color:#8fe9d5; }
[data-testid="stDataFrame"]{ border-radius:18px; overflow:hidden; }
@media (max-width:900px){ .hero-grid,.steps,.footer{ grid-template-columns:1fr; } .statrow{ grid-template-columns:1fr 1fr; } .hero .h1{ font-size:2.2rem; } }
</style>
"""


def hero_html(meta, live):
    m = meta["metrics"]
    bars = "".join(f'<span style="height:{h}%"></span>' for h in (38, 55, 44, 72, 60, 88, 70, 96))
    return f"""
<div class="hero">
  <div class="nav">
    <div class="brand"><i>&#9670;</i> UPI Fraud Shield</div>
    <div class="navlinks"><span>Predict</span><span>Model insights</span><span>Live monitoring</span><span>MLOps pipeline</span></div>
    <div style="display:flex;gap:10px"><span class="pill">Experiment 8</span><span class="pill solid">E024</span></div>
  </div>
  <div class="hero-grid">
    <div>
      <div class="chip"><b></b> {meta['best_model']} &middot; {meta['training_method']} &middot; serving live</div>
      <div class="h1">Stop UPI fraud <span>before the money moves</span></div>
      <p class="lead">Score every payment in milliseconds. Trained on 2,800 real-pattern transactions, tracked in MLflow,
      shipped with Docker &amp; Kubernetes and watched by Prometheus and Grafana.</p>
      <div class="cta"><span class="pill solid">Check a transaction</span><span class="pill">View model insights</span></div>
    </div>
    <div class="glass-stack">
      <div class="glass g1"><small>Fraud caught (by value)</small><div class="big">{meta['fraud_capture_rate']:.1f}%</div>
        <div class="mini-bars">{bars}</div></div>
      <div class="glass g2"><small>Checked so far</small><div class="big">{live['total']:,}</div><small>{live['fraud']:,} flagged as fraud</small></div>
      <div class="glass g3"><small>Decision threshold</small><div class="big">{meta['optimal_threshold']:.3f}</div><small>tuned for best F1</small></div>
    </div>
  </div>
  <div class="statrow">
    <div class="stat"><div class="v">{m['ROC-AUC']:.3f}</div><div class="l">ROC-AUC</div></div>
    <div class="stat"><div class="v">{m['F1-Macro']:.3f}</div><div class="l">F1-Macro</div></div>
    <div class="stat"><div class="v">{m['Recall']:.0%}</div><div class="l">Fraud recall</div></div>
    <div class="stat"><div class="v">14</div><div class="l">model runs compared</div></div>
  </div>
</div>"""


def section(eyebrow, headline, grey="", sub=""):
    grey_html = f" <span>{grey}</span>" if grey else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="eyebrow">{eyebrow}</div><div class="headline">{headline}{grey_html}</div>{sub_html}'


def _color(v):
    return "#19b37d" if v < 0.34 else ("#f4a62a" if v < 0.67 else "#ef4a4a")


def signals_html(signals):
    rows = "".join(
        f'<div class="sig"><span>{name}</span><div class="track"><div class="fill" style="width:{v*100:.0f}%;background:{_color(v)}"></div></div>'
        f"<b>{v*100:.0f}</b></div>" for name, v in signals)
    return (f'<div class="card-title">Risk protection <em>live signals</em></div>{rows}'
            '<div class="note">Signals update as you move the controls and are only a quick guide. '
            "The trained model gives the real fraud probability when you press Predict.</div>")


def verdict_html(result):
    p = result["fraud_probability"]
    color = "#ef4a4a" if result["is_fraud"] else "#19b37d"
    circ = 2 * 3.14159 * 54
    gauge = (f'<svg width="130" height="130" viewBox="0 0 130 130"><circle cx="65" cy="65" r="54" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="12"/>'
             f'<circle cx="65" cy="65" r="54" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
             f'stroke-dasharray="{p*circ:.1f} {circ:.1f}" transform="rotate(-90 65 65)"/>'
             f'<text x="65" y="63" text-anchor="middle" fill="#fff" font-size="26" font-weight="800">{p*100:.1f}%</text>'
             f'<text x="65" y="84" text-anchor="middle" fill="rgba(255,255,255,.6)" font-size="10">fraud probability</text></svg>')
    if result["is_fraud"]:
        badge, head, text = "fraud", "FRAUD", "Block or send this payment for review."
    else:
        badge, head, text = "legit", "LEGITIMATE", "Looks normal - safe to approve."
    return (f'<div class="card-title">Model decision</div><div class="verdict">{gauge}<div class="txt">'
            f'<span class="badge {badge}">{head}</span><div class="vt">{text}</div>'
            f'<p>Decision threshold {result["threshold"]*100:.1f}% &middot; model score {p*100:.1f}%</p></div></div>')


def recent_html(history):
    if not history:
        return ""
    rows = "".join(f'<div class="recent"><span>{h["time"]} &middot; &#8377;{h["amount"]:,.0f}</span><span>{h["label"]} &middot; {h["p"]*100:.0f}%</span></div>'
                   for h in history[:5])
    return f'<div class="card-title" style="margin-top:18px">Recent checks</div>{rows}'


def tiles_html(items):
    cells = "".join(f'<div class="tile"><div class="l">{label}</div><div class="v {cls}">{value}</div></div>' for label, value, cls in items)
    return f'<div class="tiles">{cells}</div>'


def hbars_html(items, best=None, integer=False):
    top = max(v for _, v in items) or 1
    rows = []
    for label, v in items:
        shown = f"{v:,.0f}" if integer else f"{v:.3f}"
        cls = "best" if label == best else ""
        rows.append(f'<div class="hbar {cls}"><span>{label}</span><div class="track"><div class="fill" style="width:{v / top * 100:.0f}%"></div></div>'
                    f"<b>{shown}</b></div>")
    return "".join(rows)


def steps_html():
    return """
<div class="steps">
  <div class="step green"><div class="n">1</div><div class="st">Understand the payment</div>
    <p>18 engineered features - night-time activity, new accounts, amount vs usual spend, device and IP mismatches - turn a raw UPI
    transaction into 79 model inputs.</p></div>
  <div class="step"><div class="n">2</div><div class="st">Score with CatBoost</div>
    <p>Seven models and two balancing methods were compared in MLflow. The winner is tuned to its best F1 threshold and explained with SHAP.</p></div>
  <div class="step"><div class="n">3</div><div class="st">Watch it in production</div>
    <p>Every request updates Prometheus counters and a latency histogram. Grafana shows traffic, errors and fraud share across pods.</p></div>
</div>"""


def trust_html():
    logos = ["CatBoost", "MLflow", "Streamlit", "Docker", "Kubernetes", "Prometheus", "Grafana", "GitHub Actions"]
    return '<div class="trust"><span class="t">Built and run with</span>' + "".join(f'<span class="logo">{x}</span>' for x in logos) + "</div>"


def footer_html():
    return """
<div class="footer">
  <div><div class="ft">Fuel your fraud checks with a model you can trust.</div>
    <p>End-to-end MLOps pipeline: Git &rarr; GitHub Actions &rarr; MLflow &rarr; Docker &rarr; Kubernetes &rarr; Prometheus &amp; Grafana.</p></div>
  <div><b>Project</b><ul><li>UPI Fraud Detection</li><li>MLOps Experiment 8</li><li>E024 &middot; SEM 11</li></ul></div>
  <div><b>Endpoints</b><ul><li>Metrics: :8000/metrics</li><li>Prometheus: :9090</li><li>Grafana: :3000</li></ul></div>
</div>"""
