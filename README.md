# 🚗 Accident Severity Predictor — End-to-End ML Pipeline

Predict accident **severity (1–4)** before it happens using only pre-accident features.  
Model: **LightGBM** · Deployment: **Streamlit Community Cloud** (recommended)

---

## 📁 Project Structure

```
accident_severity/
├── notebooks/
│   └── accident_severity_model.ipynb   ← full training pipeline (run in Jupyter)
├── src/
│   └── predict.py                      ← reusable inference module
├── app/
│   └── app.py                          ← Streamlit deployment app
├── artifacts/                          ← generated after running the notebook
│   ├── preprocessor.joblib
│   ├── lgbm_model.joblib
│   └── model_metadata.json
├── data/
│   └── data.csv                        ← your dataset (place here)
├── requirements.txt
└── README.md
```

---

## 🔄 Step 1 — Train the Model (Jupyter Notebook)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data
cp /path/to/data.csv data/data.csv

# 4. Open notebook
jupyter notebook notebooks/accident_severity_model.ipynb

# 5. Run ALL cells — artifacts/ will be populated
```

After running the notebook you will have:
- `artifacts/preprocessor.joblib`
- `artifacts/lgbm_model.joblib`
- `artifacts/model_metadata.json`

---

## 🚀 Step 2 — Run Streamlit Locally (VS Code)

```bash
cd accident_severity
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment Comparison

| Platform | Best for | Free tier | Notes |
|---|---|---|---|
| **Streamlit Community Cloud** ✅ | ML/data apps | Yes (generous) | 1-click deploy from GitHub, built for Streamlit |
| **Render** | General web apps | Yes (spins down) | Docker/Python support, sleeps after 15 min inactivity |
| **Railway** | Full-stack apps | $5 credit/month | Fastest cold start among free tiers |
| **Vercel / Netlify** | ❌ Not recommended | — | Frontend-only platforms; no Python runtime |

### ✅ Recommended: Streamlit Community Cloud

It is purpose-built for Streamlit apps, has a generous free tier, and deploys
directly from a GitHub repository with zero configuration.

---

## ☁️ Step 3 — Deploy to Streamlit Community Cloud

### 3a. Prepare GitHub repo

```
your-repo/
├── app.py                  ← entry point (move app/app.py here)
├── src/
│   └── predict.py
├── artifacts/
│   ├── preprocessor.joblib
│   ├── lgbm_model.joblib
│   └── model_metadata.json
└── requirements.txt
```

> ⚠️ The `artifacts/` folder with `.joblib` files **must be committed to git**.  
> If files are large, use Git LFS: `git lfs track "*.joblib"`

### 3b. Deploy

1. Push your repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch, and set **Main file path** = `app.py`
5. Click **Deploy** — done in ~2 minutes ✓

### 3c. requirements.txt for cloud (stripped-down)

```
lightgbm>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
streamlit>=1.35.0
plotly>=5.20.0
```

---

## ☁️ Alternative: Deploy to Render

```bash
# Procfile (create in root)
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect repo, set **Start command** = contents of Procfile above
4. Deploy

---

## 🧪 Features Used

| Feature | Type | Description |
|---|---|---|
| HOUR | numeric | Hour of day (0–23) |
| MINUTE | numeric | Minute (0–59) |
| HOUR_SIN / HOUR_COS | numeric | Cyclical time encoding |
| DAY_OF_WEEK | numeric | 1=Mon … 7=Sun |
| LIGHT_CONDITION | numeric | 1=Day … 9=N/A |
| SPEED_ZONE | numeric | Speed limit (km/h) |
| IS_PEAK_HOUR | numeric | 1 if 7–9am or 5–7pm |
| IS_WEEKEND | numeric | 1 if Sat/Sun |
| ROAD_GEOMETRY_DESC | categorical | T-intersection, Cross, etc. |
| HIGHWAY | categorical | Highway segment name |
| TIME_OF_DAY | categorical | Night/Morning/Afternoon/Evening |
| SPEED_RISK | categorical | LOW/MEDIUM/HIGH/VERY_HIGH |

**Excluded (post-accident leakage):** `DCA_DESC`, `ACCIDENT_TYPE_DESC`

---

## 📊 Model

- **Algorithm:** LightGBM (multiclass, num_class=4)
- **Class imbalance:** handled via `class_weight='balanced'`
- **Early stopping:** on validation set (50 rounds patience)
- **Pipeline:** `ColumnTransformer` (StandardScaler for numeric, OrdinalEncoder for categorical) → LightGBM
- **Target:** SEVERITY converted from 1–4 → 0–3 for LightGBM, then back to 1–4 at output
