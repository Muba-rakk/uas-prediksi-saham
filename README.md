# 📈 Prediksi Harga Saham PT Aneka Tambang Tbk (ANTM)
### Perbandingan LSTM, GRU, Linear Regression & Model Hybrid

*Tugas UAS Analitika Data*

**Tech stack:** Python 3.11 · TensorFlow/Keras · scikit-learn · pandas · NumPy · Plotly · Matplotlib · Seaborn · statsmodels · dikelola dengan `uv`

---

## 🧭 Tentang Proyek

Proyek ini memprediksi harga penutupan harian saham **PT Aneka Tambang Tbk (ANTM)** di Bursa Efek Indonesia menggunakan data historis dan beberapa pendekatan machine learning / deep learning, mulai dari model statistik sederhana hingga model sekuensial berbasis neural network.

Alur kerja mengikuti pipeline analitika data end-to-end: **EDA → Preprocessing → Modeling → Evaluasi → Visualisasi**, dengan empat model yang dibandingkan secara adil pada split data yang sama.

## 🗂️ Struktur Repository

```
uas-prediksi-saham/
├── data/
│   └── raw/
│       └── data_antam.csv        # Data harga historis ANTM
├── notebooks/
│   ├── 01_eda.ipynb              # Eksplorasi data, tren harga & volume, uji stasioneritas
│   ├── 02_preprocessing.ipynb    # Cleaning, fitur teknikal, normalisasi, sliding window
│   ├── 03_modeling.ipynb         # Training Linear Regression, LSTM, GRU, & Hybrid
│   ├── 04_evaluasi.ipynb         # Perhitungan metrik & analisis error
│   └── 05_visualisasi.ipynb      # Candlestick chart & forecasting interaktif
├── pyproject.toml                # Dependencies (dikelola dengan uv)
└── uv.lock
```

## 🔬 Metodologi

**1. Exploratory Data Analysis** — visualisasi tren harga & volume, distribusi, korelasi antar fitur, dan uji stasioneritas data time series.

**2. Preprocessing** — pembersihan data, koreksi skala harga, dan penambahan fitur teknikal:
- Moving Average (MA7, MA30)
- RSI (14 hari)
- Bollinger Bands (upper, lower, width)
- Split data kronologis (train/val/test) + normalisasi MinMaxScaler
- Sliding window sequence (lookback 60 hari) untuk model sekuensial

**3. Modeling** — empat model dilatih dan dibandingkan:

| Model | Deskripsi |
|---|---|
| **Linear Regression** | Baseline sederhana untuk menangkap tren linear |
| **LSTM** | Long Short-Term Memory untuk pola non-linear jangka panjang |
| **GRU** | Gated Recurrent Unit, arsitektur lebih ringan dari LSTM |
| **Hybrid (LR + GRU)** | *Sequential residual modeling* — LR menangkap tren, GRU memprediksi residualnya |

**4. Evaluasi** — RMSE, MAE, MAPE, analisis distribusi error, dan error terhadap waktu.

**5. Visualisasi** — candlestick chart interaktif, prediksi vs aktual, heatmap korelasi fitur, serta forecasting 1 hari dan 30 hari ke depan.

## 📊 Hasil Perbandingan Model

| Model | RMSE (Rp) | MAE (Rp) | MAPE |
|---|---:|---:|---:|
| Linear Regression (Baseline) | 147.24 | 111.58 | 3.26% |
| **Hybrid (LR + GRU)** | **147.37** | **111.67** | **3.26%** |
| GRU | 407.67 | 370.84 | 10.62% |
| LSTM | 568.34 | 505.31 | 14.29% |

> Menariknya, model **Linear Regression** sebagai baseline tampil setara dengan model Hybrid dan mengungguli LSTM maupun GRU murni pada dataset ini — menunjukkan tren harga ANTM pada periode data cenderung dominan linear, sementara GRU lebih baik menangkap volatilitas dibanding LSTM (RMSE lebih rendah 28% dengan parameter 24% lebih sedikit).

## ⚙️ Instalasi & Menjalankan

Proyek ini menggunakan [uv](https://github.com/astral-sh/uv) untuk manajemen dependency.

```bash
# Clone repository
git clone https://github.com/rachmankirin/uas-prediksi-saham.git
cd uas-prediksi-saham

# Install dependencies
uv sync

# Jalankan Jupyter Lab
uv run jupyter lab
```

Kemudian jalankan notebook secara berurutan dari `01_eda.ipynb` hingga `05_visualisasi.ipynb`.

**Requirements:** Python ≥ 3.11.9 · pandas · numpy · scikit-learn · tensorflow · matplotlib · seaborn · plotly · statsmodels

---

Dibuat sebagai bagian dari Tugas UAS Analitika Data
