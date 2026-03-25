## ----
## Reprezentatywność -- miary i przykłady
## Autor: Maciej Beręsewicz
## ----

## Pakiety ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

## Funkcje pomocnicze ----

def ddc(mu, muhat, N, n):
    """Data Defect Correlation (Meng 2018)."""
    f = n / N
    sigma_G = np.sqrt(mu * (1 - mu))
    rho = (muhat - mu) / (sigma_G * np.sqrt((1 - f) / f))
    return rho

def smd(x_treat, x_ctrl):
    """Standaryzowana różnica średnich."""
    return (x_treat.mean() - x_ctrl.mean()) / np.sqrt((x_treat.var() + x_ctrl.var()) / 2)

# Data Defect Index (Meng 2018) ----

## Wybory w USA 2016
## dane wyeksportowane z R: write.csv(ddi::g2016, "codes/data/g2016.csv", row.names=FALSE)
g2016 = pd.read_csv("../data/g2016.csv")
g2016[["st", "pct_djt_voters", "cces_pct_djt_vv", "tot_votes", "cces_n_vv"]].head()

## DDC dla całego USA ----
rho = ddc(
    mu    = 62984824 / 136639786,
    muhat = 12284 / 35829,
    N     = 136639786,
    n     = 35829
)
print(f"ddc: {rho:.6f}")
print(f"ddi: {rho**2:.2e}")

## DDC dla każdego stanu ----
g2016["ddc"] = ddc(
    mu    = g2016["pct_djt_voters"].values,
    muhat = g2016["cces_pct_djt_vv"].values,
    N     = g2016["tot_votes"].values,
    n     = g2016["cces_n_vv"].values
)
g2016["blad"] = g2016["cces_pct_djt_vv"] - g2016["pct_djt_voters"]
print(g2016["ddc"].describe())

## Top 10 stanów z największym |rho|
top10 = g2016.reindex(g2016["ddc"].abs().sort_values(ascending=False).index)
print(top10[["st", "pct_djt_voters", "cces_pct_djt_vv", "blad", "ddc"]].head(10))

## Wykresy ----

## Oficjalny wynik vs. szacunek CCES
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(g2016["pct_djt_voters"], g2016["cces_pct_djt_vv"],
           c="steelblue", alpha=0.7, edgecolors="none")
ax.plot([0, 1], [0, 1], "r--", lw=2, label="linia y = x")
mask = g2016["blad"].abs() > 0.05
for _, row in g2016[mask].iterrows():
    ax.annotate(row["st"], (row["pct_djt_voters"], row["cces_pct_djt_vv"]),
                fontsize=7, ha="left")
ax.set_xlabel("Oficjalny odsetek głosów na Trumpa")
ax.set_ylabel("Szacunek CCES")
ax.set_title("Oficjalny wynik vs. CCES (2016)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left")
plt.tight_layout(); plt.show()

## Błąd estymacji po stanach
g_sorted = g2016.sort_values("blad")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if b < 0 else "coral" for b in g_sorted["blad"]]
ax.bar(range(len(g_sorted)), g_sorted["blad"], color=colors)
ax.set_xticks(range(len(g_sorted)))
ax.set_xticklabels(g_sorted["st"], rotation=90, fontsize=6)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Błąd (CCES - oficjalny)")
ax.set_title("Błąd estymacji CCES po stanach")
plt.tight_layout(); plt.show()

## DDC po stanach
g_sorted2 = g2016.sort_values("ddc")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if d < 0 else "coral" for d in g_sorted2["ddc"]]
ax.bar(range(len(g_sorted2)), g_sorted2["ddc"], color=colors)
ax.set_xticks(range(len(g_sorted2)))
ax.set_xticklabels(g_sorted2["st"], rotation=90, fontsize=6)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel(r"$\hat{\rho}_{R,G}$")
ax.set_title("Data Defect Correlation po stanach")
plt.tight_layout(); plt.show()

## Paradoks big data: Z_n vs wielkość populacji ----
g2016["z_n"] = (
    (g2016["cces_pct_djt_vv"] - g2016["pct_djt_voters"]) /
    np.sqrt(g2016["cces_pct_djt_vv"] * (1 - g2016["cces_pct_djt_vv"]) / g2016["cces_n_vv"])
)

fig, ax = plt.subplots(figsize=(7, 5))
log_N = np.log10(g2016["tot_votes"])
ax.axhspan(-2, 2, color="grey", alpha=0.2, label="95% CI (|Z| ≤ 2)")
ax.scatter(log_N, g2016["z_n"], c="steelblue", alpha=0.7, edgecolors="none", zorder=3)
for _, row in g2016.iterrows():
    ax.annotate(row["st"], (np.log10(row["tot_votes"]), row["z_n"]),
                fontsize=5.5, ha="left", va="bottom")
slope, intercept = np.polyfit(log_N, g2016["z_n"], 1)
x_line = np.linspace(log_N.min(), log_N.max(), 100)
ax.plot(x_line, intercept + slope * x_line, "r--", lw=2, label="OLS")
lo = lowess(g2016["z_n"].values, log_N.values, frac=0.6)
ax.plot(lo[:, 0], lo[:, 1], color="darkgreen", lw=2, label="LOESS")
ax.axhline(0, color="grey", lw=0.5)
ax.set_xlabel(r"$\log_{10}(N_{\mathrm{stan}})$")
ax.set_ylabel(r"$Z_n$ (Trump)")
ax.set_title("Paradoks big data: błąd rośnie z wielkością populacji")
ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
plt.tight_layout(); plt.show()

## Ilustracja paradoksu big data ----
N = 38e6
rho_vals = np.linspace(0.0001, 0.006, 500)

def z_n_curve(rho, f_share):
    n = N * f_share
    D0 = (1 - f_share) / f_share
    return np.sqrt(n) * np.sqrt(D0 * rho**2 / (1 - D0 * rho**2))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rho_vals, z_n_curve(rho_vals, 0.8), "k-", lw=2, label="rejestr (80%)")
ax.plot(rho_vals, z_n_curve(rho_vals, 0.01), "r-", lw=2, label="próba losowa (1%)")
ax.axvline(0.001, ls=":", color="red", alpha=0.5)
ax.axhline(z_n_curve(0.001, 0.01), ls=":", color="red", alpha=0.5)
ax.axvline(0.005, ls=":", color="black", alpha=0.5)
ax.axhline(z_n_curve(0.005, 0.8), ls=":", color="black", alpha=0.5)
ax.set_xlabel(r"$\rho_{R,G}$")
ax.set_ylabel(r"Błąd standardowy ($Z_n$)")
ax.set_title("Paradoks Big Data (Meng 2018, N = 38 mln)")
ax.legend(loc="lower right", framealpha=0.8)
ax.set_ylim(0, 15)
plt.tight_layout(); plt.show()

# Balans zmiennych pomocniczych ----

## Dane: JVS vs. CBOP
## wyeksportowane z R:
## library(nonprobsvy); write.csv(jvs, "codes/data/jvs.csv", row.names=FALSE)
## library(nonprobsvy); write.csv(admin, "codes/data/admin.csv", row.names=FALSE)
jvs = pd.read_csv("../data/jvs.csv")
admin = pd.read_csv("../data/admin.csv")

## Porównanie struktury prób ----
tab_jvs = jvs["size"].value_counts(normalize=True).sort_index()
tab_admin = admin["size"].value_counts(normalize=True).sort_index()
print(pd.DataFrame({"JVS": tab_jvs.round(3), "CBOP": tab_admin.round(3)}))

tab_jvs_p = jvs["private"].value_counts(normalize=True).sort_index()
tab_admin_p = admin["private"].value_counts(normalize=True).sort_index()
print(pd.DataFrame({"JVS": tab_jvs_p.round(3), "CBOP": tab_admin_p.round(3)}).rename(
    index={0: "Publiczna", 1: "Prywatna"}))

## Wykresy porównania
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(tab_jvs)); w = 0.35
axes[0].bar(x - w/2, tab_jvs.values, w, color="steelblue", label="JVS")
axes[0].bar(x + w/2, tab_admin.values, w, color="coral", label="CBOP")
axes[0].set_xticks(x); axes[0].set_xticklabels(tab_jvs.index)
axes[0].set_title("Rozkład wg wielkości firmy"); axes[0].set_ylabel("Odsetek"); axes[0].legend()
x2 = np.arange(2)
axes[1].bar(x2 - w/2, tab_jvs_p.values, w, color="steelblue", label="JVS")
axes[1].bar(x2 + w/2, tab_admin_p.values, w, color="coral", label="CBOP")
axes[1].set_xticks(x2); axes[1].set_xticklabels(["Publiczna", "Prywatna"])
axes[1].set_title("Rozkład wg sektora"); axes[1].set_ylabel("Odsetek")
plt.tight_layout(); plt.show()

## SMD ----
wspol = ["private", "size", "nace", "region"]
jvs_dum = pd.get_dummies(jvs[wspol], columns=["size", "nace", "region"], dtype=float)
admin_dum = pd.get_dummies(admin[wspol], columns=["size", "nace", "region"], dtype=float)
all_cols = sorted(set(jvs_dum.columns) | set(admin_dum.columns))
jvs_dum = jvs_dum.reindex(columns=all_cols, fill_value=0)
admin_dum = admin_dum.reindex(columns=all_cols, fill_value=0)

smd_vals = {col: smd(admin_dum[col], jvs_dum[col]) for col in all_cols}
smd_df = pd.DataFrame({"SMD": smd_vals}).sort_values("SMD", key=abs, ascending=False)
print(smd_df.round(4).head(15))

## Propensity score ----
jvs_sub = jvs[wspol].copy(); jvs_sub["source"] = 0
admin_sub = admin[wspol].copy(); admin_sub["source"] = 1
combined = pd.concat([jvs_sub, admin_sub], ignore_index=True)
combined_dum = pd.get_dummies(combined[wspol], columns=["size", "nace", "region"], dtype=float)

X = sm.add_constant(combined_dum)
y = combined["source"]
ps_model = sm.Logit(y, X).fit(disp=0)
ps_hat = ps_model.predict(X)

## Wagi IPW
ps_jvs = ps_hat[y == 0].values
weights = ps_jvs / (1 - ps_jvs)

## SMD po ważeniu ----
smd_adj = {}
for col in all_cols:
    weighted_mean = np.average(jvs_dum[col], weights=weights)
    weighted_var = np.average((jvs_dum[col] - weighted_mean)**2, weights=weights)
    admin_mean = admin_dum[col].mean()
    admin_var = admin_dum[col].var()
    pooled_sd = np.sqrt((admin_var + weighted_var) / 2)
    smd_adj[col] = (admin_mean - weighted_mean) / pooled_sd if pooled_sd > 0 else 0

balance = pd.DataFrame({"SMD_unadj": smd_df["SMD"], "SMD_adj": pd.Series(smd_adj)})
balance = balance.sort_values("SMD_unadj", key=abs, ascending=False)
print(balance.round(4).head(15))

## Wykres Love'a ----
balance_sorted = balance.sort_values("SMD_unadj", key=abs, ascending=True)
fig, ax = plt.subplots(figsize=(8, max(6, len(balance_sorted) * 0.2)))
y_pos = np.arange(len(balance_sorted))
ax.scatter(balance_sorted["SMD_unadj"].abs(), y_pos, c="coral", s=40, zorder=3, label="Unadjusted")
ax.scatter(balance_sorted["SMD_adj"].abs(), y_pos, c="steelblue", s=40, zorder=3, label="Adjusted")
ax.axvline(0.1, ls="--", color="black", lw=1, alpha=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(balance_sorted.index, fontsize=6)
ax.set_xlabel("|SMD|"); ax.set_title("Balans zmiennych: CBOP vs JVS")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()

# R-indicator ----
R_indicator = 1 - 2 * ps_hat.std()
print(f"R-indicator: {R_indicator:.4f}")
print(f"SD propensity scores: {ps_hat.std():.4f}")

## Rozkład propensity scores
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ps_hat, bins=30, color="steelblue", edgecolor="white")
ax.axvline(ps_hat.mean(), ls="--", color="red", lw=2, label=f"Średnia: {ps_hat.mean():.3f}")
ax.set_xlabel("Prawdopodobieństwo przynależności do CBOP")
ax.set_ylabel("Liczba obserwacji")
ax.set_title("Rozkład propensity scores P(CBOP | X)")
ax.legend(framealpha=0.8)
plt.tight_layout(); plt.show()
