from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from pathlib import Path
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = pd.read_csv(DATA_PATH)
X, y = df[FEATURES], df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
METRICS = {
    "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
    "precision": round(precision_score(y_test, pred, average="weighted") * 100, 2),
    "recall": round(recall_score(y_test, pred, average="weighted") * 100, 2),
    "f1": round(f1_score(y_test, pred, average="weighted") * 100, 2),
}
CROPS = sorted(df["label"].unique().tolist())
STATS = {
    "rows": int(df.shape[0]),
    "features": len(FEATURES),
    "crops": len(CROPS),
    "missing": int(df.isna().sum().sum()),
    "duplicates": int(df.duplicated().sum()),
    "crop_counts": df["label"].value_counts().sort_values(ascending=False).to_dict(),
    "means": {k: round(float(v), 2) for k, v in df[FEATURES].mean().items()},
    "correlations": df[FEATURES].corr().round(2).to_dict(),
}

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;"))

def chatbot_reply(question):
    q = question.lower().strip()
    if not q:
        return "Please type a question about soil or crops."
    if any(word in q for word in ["hello", "hi", "namaste", "नमस्ते"]):
        return "Namaste! 🌱 Ask me about soil pH, NPK, rainfall or crop recommendations."
    if "ph" in q or "acidic" in q or "alkaline" in q or "pH" in question:
        return ("Soil pH measures acidity or alkalinity. pH 7 is neutral, below 7 is acidic, "
                "and above 7 is alkaline. Crop suitability also depends on nutrients, climate "
                "and local conditions.")
    if any(word in q for word in ["npk", "nitrogen", "phosphorus", "potassium"]):
        return ("NPK stands for Nitrogen, Phosphorus and Potassium. These nutrients support "
                "plant growth. The right amount depends on the crop and soil-test results.")
    if any(word in q for word in ["rainfall", "rain", "rainy", "barish", "baarish"]):
        return ("Rainfall requirements vary by crop, season and soil drainage. Consider irrigation "
                "and local climate before choosing a crop.")
    if any(word in q for word in ["temperature", "humidity", "moisture", "weather"]):
        return ("Temperature and humidity affect crop growth. Use local weather information along "
                "with soil-test results before making a farming decision.")
    if any(word in q for word in ["crop", "recommend", "fasal", "which plant", "which crop"]):
        return (f"Our dataset contains {len(CROPS)} crop categories. Enter your N, P, K, temperature, "
                "humidity, pH and rainfall values in the Analyze Soil form to get a model prediction.")
    if any(word in q for word in ["thank", "thanks", "shukriya", "dhanyawad"]):
        return "You're welcome! 🌱 Happy farming."
    return ("I can answer basic questions about soil pH, NPK nutrients, rainfall, temperature, "
            "humidity and crop recommendations. Try asking about one of these topics.")

def page(message="", values=None):
    values = values or {"N":"90","P":"42","K":"43","temperature":"25","humidity":"80","ph":"6.5","rainfall":"200"}
    fields = [
        ("N","Nitrogen (N)","0","140","90"), ("P","Phosphorus (P)","0","145","42"),
        ("K","Potassium (K)","0","205","43"), ("temperature","Temperature (°C)","-10","60","25"),
        ("humidity","Humidity (%)","0","100","80"), ("ph","Soil pH","0","14","6.5"),
        ("rainfall","Rainfall (mm)","0","500","200")
    ]
    controls = "".join(
        f'<label>{label}<input required type="number" step="any" name="{key}" min="{lo}" max="{hi}" value="{esc(values.get(key,default))}"></label>'
        for key,label,lo,hi,default in fields
    )
    metric_cards = "".join(f'<div class="metric"><span>{label}</span><strong>{value}%</strong></div>' for label,value in [
        ("Accuracy",METRICS["accuracy"]),("Precision",METRICS["precision"]),("Recall",METRICS["recall"]),("F1 score",METRICS["f1"])])
    means = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k,v in STATS["means"].items())
    crop_rows = "".join(f"<tr><td>{esc(crop)}</td><td>{count}</td></tr>" for crop,count in sorted(STATS["crop_counts"].items()))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SoilSense | Crop Recommendation</title><link rel="stylesheet" href="/style.css">
<style>
.chatbot{{position:fixed;right:20px;bottom:20px;z-index:9999;font-family:Arial,sans-serif}}
.chat-toggle{{border:0;border-radius:999px;padding:14px 18px;background:#087f5b;color:white;font-weight:bold;box-shadow:0 5px 18px #0003;cursor:pointer}}
.chat-window{{display:none;width:min(340px,calc(100vw - 32px));height:420px;background:#fff;color:#18352a;border:1px solid #d5e5dc;border-radius:18px;box-shadow:0 8px 32px #0003;margin-bottom:10px;overflow:hidden}}
.chat-window.open{{display:flex;flex-direction:column}}
.chat-head{{background:#087f5b;color:#fff;padding:14px;font-weight:bold;display:flex;justify-content:space-between;align-items:center}}
.chat-head button{{border:0;background:transparent;color:white;font-size:20px;cursor:pointer}}
.chat-messages{{padding:12px;overflow-y:auto;flex:1;background:#f6fbf7;display:flex;flex-direction:column;gap:9px}}
.chat-msg{{max-width:88%;padding:10px 12px;border-radius:13px;font-size:14px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere}}
.chat-msg.bot{{align-self:flex-start;background:#e4f2e9}}
.chat-msg.user{{align-self:flex-end;background:#087f5b;color:#fff}}
.chat-form{{display:flex;gap:7px;padding:10px;border-top:1px solid #e2e9e4}}
.chat-form input{{min-width:0;flex:1;border:1px solid #cbd9d0;border-radius:10px;padding:10px;font-size:14px}}
.chat-form button{{border:0;border-radius:10px;background:#087f5b;color:#fff;padding:0 13px;cursor:pointer}}
.chat-note{{font-size:11px;color:#66756c;text-align:center;margin:0 8px 8px}}
</style></head><body>
<header><div class="brand">🌱 SoilSense <span class="brand-by">by Sumit</span></div>
<nav><a href="#analyze">Analyze soil</a><a href="#statistics">Statistics</a><a href="#model">Model</a></nav></header>
<main>
<section class="hero"><div><p class="eyebrow">DATA-DRIVEN AGRICULTURE</p><h1>Know your soil.<br><em>Explore the right crop.</em></h1>
<p class="intro">Enter soil nutrients and climate values to get a machine-learning recommendation based on the training dataset.</p>
<a class="cta" href="#analyze">Analyze my soil ↓</a></div><div class="hero-art" aria-hidden="true"><div class="sun"></div><div class="hill hill1"></div><div class="hill hill2"></div><div class="plant">🌿</div></div></section>
<section class="stats-strip"><div><strong>{STATS["rows"]:,}</strong><span>dataset records</span></div><div><strong>{STATS["crops"]}</strong><span>crop categories</span></div><div><strong>{STATS["missing"]}</strong><span>missing values</span></div><div><strong>{STATS["duplicates"]}</strong><span>duplicate rows</span></div></section>
<section id="analyze" class="section"><div class="section-heading"><p class="eyebrow">01 / PREDICTION</p><h2>Enter soil conditions</h2><p>Use measured values where possible. Example values are prefilled so you can test the website.</p></div>
<form method="post" action="/predict" class="form-card"><div class="fields">{controls}</div><button type="submit">Recommend a crop <span>→</span></button><p class="fineprint">Predictions are educational estimates, not a substitute for local agronomic advice.</p></form>
{message}</section>
<section id="statistics" class="section tinted"><div class="section-heading"><p class="eyebrow">02 / DATA EXPLORATION</p><h2>Inside the dataset</h2><p>The uploaded CSV was checked before model training.</p></div>
<div class="two-col"><div class="panel"><h3>Records per crop</h3><div class="bars">{''.join(f'<div class="bar-row"><span>{esc(k)}</span><div class="bar-track"><div class="bar-fill" style="width:{v/max(STATS["crop_counts"].values())*100:.1f}%"></div></div><b>{v}</b></div>' for k,v in sorted(STATS["crop_counts"].items()))}</div></div>
<div class="panel"><h3>Average input values</h3><table><thead><tr><th>Feature</th><th>Mean</th></tr></thead><tbody>{means}</tbody></table><p class="fineprint">Means are descriptive summaries of this dataset, not universal soil targets.</p></div></div>
</section>
<section id="model" class="section"><div class="section-heading"><p class="eyebrow">03 / MODEL EVALUATION</p><h2>Test-set performance</h2><p>Random Forest trained on 80% of records and evaluated on a held-out 20% test set. Stratified split, random_state=42.</p></div>
<div class="metrics">{metric_cards}</div><div class="panel note"><h3>How to interpret this</h3><p>High test accuracy shows how well the model predicts labels in this dataset. It does not prove a crop will succeed on a particular farm, because local soil testing, season, irrigation, pests and regional conditions also matter.</p></div>
</section>
<footer><div class="brand">🌱 SoilSense</div><p>Student machine-learning project · Built with Python, pandas and scikit-learn</p><p class="made-by">Made with <span aria-label="love">♥</span> by Sumit</p></footer>
</main>
<div class="chatbot">
  <div id="chatWindow" class="chat-window" aria-label="SoilSense farming chatbot">
    <div class="chat-head"><span>🌱 SoilSense Assistant</span><button type="button" id="chatClose" aria-label="Close chat">×</button></div>
    <div id="chatMessages" class="chat-messages"><div class="chat-msg bot">Namaste! I can help with soil pH, NPK, rainfall and crop recommendations. Ask me a question.</div></div>
    <form id="chatForm" class="chat-form"><input id="chatInput" type="text" maxlength="500" placeholder="Ask about soil or crops..." autocomplete="off" required><button type="submit">Send</button></form>
    <p class="chat-note">Basic educational assistant · Check advice with local experts.</p>
  </div>
  <button type="button" id="chatToggle" class="chat-toggle">💬 Ask SoilSense</button>
</div>
<script>
(function() {{
  const win = document.getElementById('chatWindow');
  const toggle = document.getElementById('chatToggle');
  const close = document.getElementById('chatClose');
  const form = document.getElementById('chatForm');
  const input = document.getElementById('chatInput');
  const messages = document.getElementById('chatMessages');
  function addMessage(text, who) {{
    const bubble = document.createElement('div');
    bubble.className = 'chat-msg ' + who;
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }}
  toggle.addEventListener('click', function() {{
    win.classList.toggle('open');
    if (win.classList.contains('open')) input.focus();
  }});
  close.addEventListener('click', function() {{ win.classList.remove('open'); }});
  form.addEventListener('submit', async function(event) {{
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    addMessage(question, 'user');
    input.value = '';
    const waiting = addMessage('Thinking...', 'bot');
    try {{
      const response = await fetch('/chat', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'}},
        body: new URLSearchParams({{message: question}})
      }});
      const data = await response.json();
      waiting.textContent = data.reply || 'Sorry, I could not answer that.';
    }} catch (error) {{
      waiting.textContent = 'Could not connect. Please try again.';
    }}
    messages.scrollTop = messages.scrollHeight;
  }});
}})();
</script>
</body></html>"""

class Handler(BaseHTTPRequestHandler):
    def send_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_html(page())
        elif self.path == "/style.css":
            try:
                body = (BASE / "style.css").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self.send_error(404)
        elif self.path == "/health":
            self.send_json({"status": "ok", "rows": STATS["rows"], "metrics": METRICS})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_data = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw_data)

        if self.path == "/chat":
            question = data.get("message", [""])[0][:500]
            self.send_json({"reply": chatbot_reply(question)})
            return

        if self.path != "/predict":
            self.send_error(404)
            return

        values = {k: data.get(k, [""])[0] for k in FEATURES}
        try:
            numbers = {k: float(v) for k, v in values.items()}
            if any(not (-10000 < v < 10000) for v in numbers.values()):
                raise ValueError("Please enter reasonable numeric values.")
            pred_crop = model.predict(pd.DataFrame([numbers], columns=FEATURES))[0]
            probs = model.predict_proba(pd.DataFrame([numbers], columns=FEATURES))[0]
            confidence = float(max(probs) * 100)
            msg = (f'<div class="result"><p class="eyebrow">MODEL RECOMMENDATION</p>'
                   f'<h3>Suggested crop: <em>{esc(pred_crop.title())}</em></h3>'
                   f'<p>Model confidence for this prediction: <strong>{confidence:.1f}%</strong></p>'
                   f'<p class="fineprint">This confidence is not a probability of real-world farming success.</p></div>')
        except Exception:
            msg = '<div class="result error"><h3>Could not process those values</h3><p>Please enter a valid number in every field.</p></div>'
        self.send_html(page(msg, values))

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer((host, port), Handler)
    print(f"SoilSense website running on {host}:{port}")
    print("Dataset:", STATS["rows"], "records | crops:", STATS["crops"])
    print("Random Forest test accuracy:", METRICS["accuracy"], "%")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWebsite stopped.")
        server.server_close()
