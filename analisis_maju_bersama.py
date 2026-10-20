"""
Studi Kasus: Mengapa Laba PT Maju Bersama Turun Saat Pendapatan Naik?
Kerangka: CRISP-DM | Level analisis: deskriptif, diagnostik, prediktif, preskriptif
Seluruh data dibangkitkan secara sintetik.

Menjalankan:
    pip install -r requirements.txt
    python analisis_maju_bersama.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, classification_report)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

rng = np.random.default_rng(42)
OUT = "output"
os.makedirs(OUT, exist_ok=True)


def judul(teks):
    print("\n" + "=" * 70 + f"\n{teks}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# BAGIAN 1: Pemahaman bisnis + generator data sintetik
# ---------------------------------------------------------------------------
HIPOTESIS = {
    "H1": "Margin turun karena diskon makin besar",
    "H2": "Penagihan melambat sehingga piutang naik",
    "H3": "Pelanggan dengan tempo panjang/diskon besar/risiko tinggi lebih sering telat bayar",
}


def generate_data(n_cust=300, n_inv=12000):
    cust = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_cust + 1)],
        "segment": rng.choice(["Retail", "Grosir", "Korporat"], n_cust, p=[.5, .3, .2]),
        "region": rng.choice(["Sumatera", "Jawa", "Kalimantan", "Sulawesi"], n_cust,
                             p=[.25, .45, .15, .15]),
        "risk_score": rng.beta(2, 8, n_cust),
    })

    inv = pd.DataFrame({
        "invoice_id": [f"INV{i:06d}" for i in range(1, n_inv + 1)],
        "customer_id": rng.choice(cust.customer_id, n_inv),
    })
    # Transaksi makin banyak di 2024 (pendapatan naik)
    days = np.arange(731)
    w = np.linspace(1, 1.8, 731)
    w /= w.sum()
    inv["invoice_date"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.choice(days, n_inv, p=w), unit="D")
    inv = inv.merge(cust, on="customer_id", how="left")

    n = len(inv)
    t = ((inv.invoice_date - pd.Timestamp("2023-01-01")).dt.days / 730).values
    inv["payment_terms"] = rng.choice([14, 30, 45, 60], n, p=[.15, .45, .25, .15])
    inv["qty"] = rng.integers(1, 200, n)
    inv["unit_price"] = rng.lognormal(10.5, 0.4, n).round(-2)
    inv["discount_pct"] = np.clip(rng.normal(5, 3, n) + 6 * t, 0, 30).round(1)

    gross = inv.qty * inv.unit_price
    inv["revenue"] = gross * (1 - inv.discount_pct / 100)
    inv["cogs"] = gross * (0.72 + rng.normal(0, 0.02, n))

    # Perilaku bayar: dipengaruhi risiko, tempo, waktu, diskon, nilai tagihan
    logit = (-4.7 + 3 * inv.risk_score + 0.02 * inv.payment_terms + 0.8 * t
             + 0.04 * inv.discount_pct + 0.3 * np.log(inv.revenue / 1e6))
    p_late = 1 / (1 + np.exp(-logit))
    late = rng.random(n) < p_late
    due = inv.invoice_date + pd.to_timedelta(inv.payment_terms, unit="D")
    offset = np.where(late, rng.integers(1, 45, n), -rng.integers(0, 10, n))
    inv["paid_date"] = due + pd.to_timedelta(offset, unit="D")
    return cust, inv


def inject_dirt(df):
    """Menyuntikkan data kotor seperti di dunia nyata."""
    df = df.copy()
    n = len(df)

    def pick(p):
        return rng.choice(n, int(n * p), replace=False)

    # Nilai kosong
    df.loc[pick(.03), "region"] = np.nan

    # Kategori tidak konsisten (huruf besar/kecil, spasi, salah eja)
    def kotori(x):
        if pd.isna(x) or rng.random() > .15:
            return x
        opsi = [x.lower(), x.upper() + " "]
        if x == "Sumatera":
            opsi.append("Sumatra")
        return opsi[rng.integers(len(opsi))]
    df["region"] = df["region"].map(kotori)

    i = pick(.003); df.loc[i, "qty"] = -df.loc[i, "qty"]                         # qty negatif
    i = pick(.003); df.loc[i, "paid_date"] = df.loc[i, "invoice_date"] - pd.Timedelta(days=3)  # bayar sebelum faktur
    i = pick(.005); df.loc[i, "discount_pct"] = rng.uniform(40, 60, len(i))       # diskon tak wajar
    df = pd.concat([df, df.sample(int(n * .01), random_state=1)])                # duplikasi
    return df.reset_index(drop=True)


cust, inv_clean = generate_data()
inv_raw = inject_dirt(inv_clean)
judul("BAGIAN 1: Data mentah dibangkitkan")
for k, v in HIPOTESIS.items():
    print(f"{k}: {v}")
print("Ukuran data mentah:", inv_raw.shape)


# ---------------------------------------------------------------------------
# BAGIAN 2: Pemahaman data
# ---------------------------------------------------------------------------
judul("BAGIAN 2: Pemahaman data")
inv_raw.info()
print("\nNilai kosong:\n", inv_raw.isna().sum()[lambda s: s > 0])
print("\nKategori region (belum konsisten):\n", inv_raw.region.value_counts(dropna=False))
print("\nInvoice duplikat:", inv_raw.invoice_id.duplicated().sum())
print("Qty <= 0:", (inv_raw.qty <= 0).sum())
print("Bayar sebelum tanggal faktur:", (inv_raw.paid_date < inv_raw.invoice_date).sum())
print("Diskon > 30% (melanggar kebijakan):", (inv_raw.discount_pct > 30).sum())
print(inv_raw.describe().T)


# ---------------------------------------------------------------------------
# BAGIAN 3: Cek granularitas sebelum join
# ---------------------------------------------------------------------------
judul("BAGIAN 3: Jebakan join dan granularitas")
assert cust.customer_id.is_unique, "Granularitas tabel pelanggan salah!"

kontak = pd.concat([cust[["customer_id"]]] * 3)  # 3 kontak per pelanggan
salah = inv_raw.merge(kontak, on="customer_id")
print(f"Sebelum join: {len(inv_raw):,} baris | sesudah join salah: {len(salah):,} baris "
      f"(x{len(salah) / len(inv_raw):.0f}) -> total pendapatan ikut tergandakan!")


# ---------------------------------------------------------------------------
# BAGIAN 4: Persiapan data
# ---------------------------------------------------------------------------
def prepare(df, cust):
    log = {"baris_awal": len(df)}
    df = df.drop_duplicates(subset="invoice_id").copy()
    log["duplikat_dibuang"] = log["baris_awal"] - len(df)

    # Standarisasi kategori
    mapping = {"jawa": "Jawa", "sumatera": "Sumatera", "sumatra": "Sumatera",
               "kalimantan": "Kalimantan", "sulawesi": "Sulawesi"}
    df["region"] = df["region"].str.strip().str.lower().map(mapping)
    # Isi nilai kosong dari master pelanggan (sumber lebih tepercaya)
    master = cust.set_index("customer_id")["region"]
    df["region"] = df["region"].fillna(df["customer_id"].map(master))

    # Transaksi mustahil secara logika
    mustahil = (df.qty <= 0) | (df.unit_price <= 0) | (df.paid_date < df.invoice_date)
    log["transaksi_mustahil_dibuang"] = int(mustahil.sum())
    df = df[~mustahil]

    # Anomali aturan bisnis (pengetahuan domain: maksimum diskon 30%)
    anomali = df.discount_pct > 30
    log["diskon_anomali_dibuang"] = int(anomali.sum())
    df = df[~anomali]

    # Fitur turunan + target
    df["due_date"] = df.invoice_date + pd.to_timedelta(df.payment_terms, unit="D")
    df["days_late"] = (df.paid_date - df.due_date).dt.days
    df["is_late"] = (df.days_late > 0).astype(int)
    df["gross_profit"] = df.revenue - df.cogs
    df["quarter"] = df.invoice_date.dt.to_period("Q").astype(str)
    log["baris_akhir"] = len(df)
    return df.reset_index(drop=True), log


judul("BAGIAN 4: Persiapan data")
df, log = prepare(inv_raw, cust)
print(log)
print(df.region.value_counts())
print(f"\nProporsi terlambat: {df.is_late.mean():.1%}  (data tidak seimbang)")


# ---------------------------------------------------------------------------
# BAGIAN 5: Analisis deskriptif dan diagnostik
# ---------------------------------------------------------------------------
judul("BAGIAN 5: Analisis deskriptif dan diagnostik")
q = df.groupby("quarter").agg(
    pendapatan=("revenue", "sum"),
    laba_kotor=("gross_profit", "sum"),
    rata2_diskon=("discount_pct", "mean"),
    tingkat_telat=("is_late", "mean"),
)
q["margin_%"] = q.laba_kotor / q.pendapatan * 100
print(q.round(2))
q.round(2).to_csv(f"{OUT}/1_deskriptif_per_kuartal.csv")


def piutang(tgl):
    m = (df.invoice_date <= tgl) & (df.paid_date > tgl)
    return df.loc[m, "revenue"].sum()


p23 = piutang(pd.Timestamp("2023-12-31"))
p24 = piutang(pd.Timestamp("2024-12-31"))
print(f"\nPiutang akhir 2023: {p23:,.0f}\nPiutang akhir 2024: {p24:,.0f}"
      f"\nKenaikan: {(p24 / p23 - 1):.0%}")

print("\nTelat bayar per tempo (hari):\n", df.groupby("payment_terms").is_late.mean().round(3))
print("\nTelat bayar per segmen:\n", df.groupby("segment").is_late.mean().round(3))
df["bin_diskon"] = pd.cut(df.discount_pct, [-1, 5, 10, 15, 20, 30])
print("\nTelat bayar per kelompok diskon:\n",
      df.groupby("bin_diskon", observed=True).is_late.mean().round(3))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
q["margin_%"].plot(ax=ax[0], marker="o", title="Margin kotor (%) per kuartal (H1)")
q["rata2_diskon"].plot(ax=ax[1], marker="o", color="orange", title="Rata-rata diskon (%)")
q["tingkat_telat"].plot(ax=ax[2], marker="o", color="red", title="Tingkat telat bayar (H2)")
for a in ax:
    a.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/1_deskriptif.png", dpi=120)

# Grafik diagnostik: tingkat telat per segmen, termin, kelompok diskon, wilayah
fig2, ax2 = plt.subplots(2, 2, figsize=(11, 8))
for a, (kol, jdl) in zip(ax2.ravel(), [("segment", "Per segmen"),
                                        ("payment_terms", "Per termin (hari)"),
                                        ("bin_diskon", "Per kelompok diskon (%)"),
                                        ("region", "Per wilayah")]):
    df.groupby(kol, observed=True).is_late.mean().mul(100).plot.bar(ax=a, color="steelblue", title=f"Telat bayar (%) - {jdl}")
    a.set_xlabel("")
    a.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/2_diagnostik.png", dpi=120)
print(f"\nGrafik disimpan di folder {OUT}/")


# ---------------------------------------------------------------------------
# BAGIAN 6: Analisis prediktif (akurasi menyesatkan & data leakage)
# ---------------------------------------------------------------------------
judul("BAGIAN 6: Analisis prediktif")
fitur_kat = ["segment", "region"]
fitur_num = ["payment_terms", "discount_pct", "revenue", "risk_score"]
FITUR = fitur_kat + fitur_num

# Split berdasarkan waktu (lebih realistis daripada acak)
cutoff = pd.Timestamp("2024-07-01")
train, test = df[df.invoice_date < cutoff], df[df.invoice_date >= cutoff]
X_tr, y_tr, X_te, y_te = train[FITUR], train.is_late, test[FITUR], test.is_late


HASIL = []


def evaluasi(nama, model, Xte=X_te):
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_te, prob)
    HASIL.append({"model": nama, "akurasi": accuracy_score(y_te, pred),
                  "recall": recall_score(y_te, pred, zero_division=0),
                  "presisi": precision_score(y_te, pred, zero_division=0), "auc": auc})
    print(f"{nama:<28} akurasi={accuracy_score(y_te, pred):.3f} "
          f"recall={recall_score(y_te, pred, zero_division=0):.3f} "
          f"presisi={precision_score(y_te, pred, zero_division=0):.3f} AUC={auc:.3f}")


# 1) Baseline: selalu menebak "tidak telat" -> akurasi tinggi, recall 0
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
evaluasi("Baseline (selalu 'tepat')", dummy)

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kat),
    ("num", StandardScaler(), fitur_num),
])

# 2) Regresi logistik dengan penanganan data tidak seimbang
logreg = Pipeline([("prep", prep),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(X_tr, y_tr)
evaluasi("Logistic Regression", logreg)

# 3) Random forest
rf = Pipeline([("prep", prep),
               ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                              min_samples_leaf=10, random_state=42, n_jobs=-1))]).fit(X_tr, y_tr)
evaluasi("Random Forest", rf)

print("\nLaporan detail Random Forest:\n", classification_report(y_te, rf.predict(X_te), digits=3))

# 4) Demo data leakage: days_late baru diketahui SETELAH pembayaran terjadi
prep_bocor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kat),
    ("num", StandardScaler(), fitur_num + ["days_late"]),
])
bocor = Pipeline([("prep", prep_bocor),
                  ("clf", LogisticRegression(max_iter=1000))]).fit(train[FITUR + ["days_late"]], y_tr)
evaluasi("Model BOCOR (days_late)", bocor, test[FITUR + ["days_late"]])
print("-> Skor nyaris sempurna itu palsu: fitur tidak tersedia saat prediksi dibuat.")

imp = pd.Series(rf.named_steps["clf"].feature_importances_,
                index=rf.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False)
print("\nTop fitur:\n", imp.head(6).round(3))
pd.DataFrame(HASIL).round(3).to_csv(f"{OUT}/3_evaluasi_model.csv", index=False)
plt.figure(figsize=(7, 5))
imp.head(10)[::-1].plot.barh(color="seagreen", title="Feature importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUT}/3_feature_importance.png", dpi=120)


# ---------------------------------------------------------------------------
# BAGIAN 7: Analisis preskriptif
# ---------------------------------------------------------------------------
judul("BAGIAN 7: Analisis preskriptif")
best = rf
test = test.copy()
test["p_telat"] = best.predict_proba(test[FITUR])[:, 1]
test["eksposur"] = test.p_telat * test.revenue

prioritas = (test.sort_values("eksposur", ascending=False)
             .head(20)[["invoice_id", "customer_id", "revenue", "payment_terms",
                        "discount_pct", "p_telat", "eksposur"]])
print("20 invoice prioritas penagihan:\n", prioritas.round(2).to_string(index=False))


def tindakan(p):
    if p >= 0.5:
        return "Hubungi sebelum jatuh tempo + eskalasi ke manajer"
    if p >= 0.3:
        return "Kirim pengingat H-7 dan H-1"
    return "Penagihan standar"


test["tindakan"] = test.p_telat.map(tindakan)
print("\n", test.tindakan.value_counts())
(test.sort_values("eksposur", ascending=False)
 [["invoice_id", "customer_id", "invoice_date", "revenue", "payment_terms",
   "discount_pct", "p_telat", "eksposur", "tindakan"]]
 .round(3).to_csv(f"{OUT}/4_rekomendasi_tindakan.csv", index=False))

# Simulasi batas diskon (asumsi: volume penjualan tidak berubah)
print("\nSimulasi kebijakan batas diskon:")
for cap in [30, 20, 15, 10]:
    alt = test.copy()
    alt["discount_pct"] = alt.discount_pct.clip(upper=cap)
    alt["revenue"] = alt.qty * alt.unit_price * (1 - alt.discount_pct / 100)
    margin = (alt.revenue.sum() - alt.cogs.sum()) / alt.revenue.sum() * 100
    p_telat = best.predict_proba(alt[FITUR])[:, 1].mean()
    print(f"  Batas {cap:>2}% -> margin {margin:5.1f}% | prediksi tingkat telat {p_telat:.1%}")

print("\nCatatan: simulasi hanya mendukung keputusan; keputusan akhir tetap di tangan manusia "
      "(misalnya risiko kehilangan pelanggan jika diskon dipangkas).")

plt.show()
