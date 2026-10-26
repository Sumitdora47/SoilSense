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
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

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
    crop_rows = "".join(f"<tr><td>{esc(crop)}</td><td>{count}</td></tr>" for crop,count in sorted(STATS["crop_counts"].items()))
    metric_cards = "".join(f'<div class="metric"><span>{label}</span><strong>{value}%</strong></div>' for label,value in [
        ("Accuracy",METRICS["accuracy"]),("Precision",METRICS["precision"]),("Recall",METRICS["recall"]),("F1 score",METRICS["f1"])])
    means = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k,v in STATS["means"].items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SoilSense | Crop Recommendation</title><link rel="stylesheet" href="/style.css"></head><body>
<header><div class="brand">
  🌱 SoilSense <span class="brand-by">by Sumit</span>
</div><nav><a href="#analyze">Analyze soil</a><a href="#statistics">Statistics</a><a href="#model">Model</a></nav></header>
<main>
<section class="hero"><div><p class="eyebrow">DATA-DRIVEN AGRICULTURE</p><h1>Know your soil.<br><em>Explore the right crop.</em></h1>
<p class="intro">Enter soil nutrients and climate values to get a machine-learning recommendation based on the training dataset.</p>
<a class="cta" href="#analyze">Analyze my soil ↓</a></div><div class="hero-art" aria-hidden="true"><div class="sun"></div><div class="hill hill1"></div><div class="hill hill2"></div><div class="plant">🌿</div></div></section>
<section class="stats-strip"><div><strong>{STATS["rows"]:,}</strong><span>dataset records</span></div><div><strong>{STATS["crops"]}</strong><span>crop categories</span></div><div><strong>{STATS["missing"]}</strong><span>missing values</span></div><div><strong>{STATS["duplicates"]}</strong><span>duplicate rows</span></div></section>
<section id="analyze" class="section"><div class="section-heading"><p class="eyebrow">01 / PREDICTION</p><h2>Enter soil conditions</h2><p>Use measured values where possible. Example values are prefilled so you can test the website.</p></div>
<form method="post" action="/predict" class="form-card"><div class="fields">{controls}</div><button type="submit">Recommend a crop <span>→</span></button><p class="fineprint">Predictions are educational estimates, not a substitute for local agronomic advice.</p></form>
{message}
</section>
<section id="statistics" class="section tinted"><div class="section-heading"><p class="eyebrow">02 / DATA EXPLORATION</p><h2>Inside the dataset</h2><p>The uploaded CSV was checked before model training.</p></div>
<div class="two-col"><div class="panel"><h3>Records per crop</h3><div class="bars">{''.join(f'<div class="bar-row"><span>{esc(k)}</span><div class="bar-track"><div class="bar-fill" style="width:{v/max(STATS["crop_counts"].values())*100:.1f}%"></div></div><b>{v}</b></div>' for k,v in sorted(STATS["crop_counts"].items()))}</div></div>
<div class="panel"><h3>Average input values</h3><table><thead><tr><th>Feature</th><th>Mean</th></tr></thead><tbody>{means}</tbody></table><p class="fineprint">Means are descriptive summaries of this dataset, not universal soil targets.</p></div></div>
</section>
<section id="model" class="section"><div class="section-heading"><p class="eyebrow">03 / MODEL EVALUATION</p><h2>Test-set performance</h2><p>Random Forest trained on 80% of records and evaluated on a held-out 20% test set. Stratified split, random_state=42.</p></div>
<div class="metrics">{metric_cards}</div><div class="panel note"><h3>How to interpret this</h3><p>High test accuracy shows how well the model predicts labels in this dataset. It does not prove a crop will succeed on a particular farm, because local soil testing, season, irrigation, pests and regional conditions also matter.</p></div>
</section>
<footer><div class="brand">🌱 SoilSense</div><p>Student machine-learning project · Built with Python, pandas and scikit-learn</p><p class="made-by">Made with <span aria-label="love">♥</span> by Sumit</p></footer>
</main></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def send_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/":
            self.send_html(page())
        elif self.path == "/style.css":
            try:
                body = (BASE/"style.css").read_bytes()
                self.send_response(200); self.send_header("Content-Type","text/css; charset=utf-8")
                self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
            except OSError:
                self.send_error(404)
        elif self.path == "/health":
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.end_headers(); self.wfile.write(json.dumps({"status":"ok","rows":STATS["rows"],"metrics":METRICS}).encode())
        else:
            self.send_error(404)
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length","0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8"))
        values = {k: data.get(k,[""])[0] for k in FEATURES}
        try:
            numbers = {k: float(v) for k,v in values.items()}
            if any(not (-10000 < v < 10000) for v in numbers.values()):
                raise ValueError("Please enter reasonable numeric values.")
            pred_crop = model.predict(pd.DataFrame([numbers], columns=FEATURES))[0]
            probs = model.predict_proba(pd.DataFrame([numbers], columns=FEATURES))[0]
            confidence = float(max(probs)*100)
            msg = (f'<div class="result"><p class="eyebrow">MODEL RECOMMENDATION</p>'
                   f'<h3>Suggested crop: <em>{esc(pred_crop.title())}</em></h3>'
                   f'<p>Model confidence for this prediction: <strong>{confidence:.1f}%</strong></p>'
                   f'<p class="fineprint">This confidence is not a probability of real-world farming success.</p></div>')
        except Exception:
            msg = '<div class="result error"><h3>Could not process those values</h3><p>Please enter a valid number in every field.</p></div>'
        self.send_html(page(msg,values))
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
        print("\\nWebsite stopped.")
        server.server_close()
