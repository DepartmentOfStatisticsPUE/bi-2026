## ----
## Wstęp do IPW
## Autor: Maciej Beręsewicz
## ----

## Pakiety ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

## Dane ----
## wyeksportowane z R:
## library(nonprobsvy); write.csv(jvs, "codes/data/jvs.csv", row.names=FALSE)
## library(nonprobsvy); write.csv(admin, "codes/data/admin.csv", row.names=FALSE)
jvs = pd.read_csv("../data/jvs.csv")
admin = pd.read_csv("../data/admin.csv")

print("admin (próba nielosowa S_A):")
print(admin.head())
print(f"\njvs (próba losowa S_B):")
print(jvs.head())

## Zmienne wspólne ----
wspol = ["private", "size", "nace", "region"]

## Funkcje pomocnicze ----
def fit_ps_model(admin, jvs, formula_vars):
    """Estymacja propensity score z pseudo-likelihood (GLM z wagami JVS)."""
    jvs_sub = jvs[formula_vars].copy()
    jvs_sub["source"] = 0
    jvs_sub["weight"] = jvs["weight"].values

    admin_sub = admin[formula_vars].copy()
    admin_sub["source"] = 1
    admin_sub["weight"] = 1.0

    combined = pd.concat([jvs_sub, admin_sub], ignore_index=True)
    combined_dum = pd.get_dummies(combined[formula_vars],
                                  columns=[v for v in formula_vars if v != "private"],
                                  dtype=float, drop_first=True)

    X = sm.add_constant(combined_dum)
    y = combined["source"]
    model = sm.GLM(y, X, family=sm.families.Binomial(),
                   freq_weights=combined["weight"]).fit()
    ps_hat = model.predict(X)

    n_jvs = (y == 0).sum()
    n_admin = (y == 1).sum()
    ps_admin = ps_hat[y == 1].values
    ps_jvs = ps_hat[y == 0].values

    return model, ps_admin, ps_jvs, X, y


def ipw_estimate(admin, jvs, formula_vars, target_var="single_shift"):
    """Estymator IPW średniej zmiennej celu."""
    model, ps_admin, ps_jvs, X, y = fit_ps_model(admin, jvs, formula_vars)

    ## wagi IPW: 1/pi(x)
    ipw_weights = 1.0 / ps_admin

    ## estymator IPW (Hajek -- znormalizowany)
    y_admin = admin[target_var].values
    mu_hat = np.average(y_admin, weights=ipw_weights)
    N_hat = ipw_weights.sum()

    return {
        "mean": mu_hat,
        "N_hat": N_hat,
        "model": model,
        "ps_admin": ps_admin,
        "ps_jvs": ps_jvs,
        "ipw_weights": ipw_weights
    }


def check_balance(admin, jvs, var, ipw_weights):
    """Sprawdzenie balansu dla zmiennej kategorycznej."""
    ## ważone proporcje w admin (po IPW)
    if admin[var].dtype == object or admin[var].nunique() <= 10:
        cats = sorted(admin[var].unique())
        result = []
        for c in cats:
            p_admin_raw = (admin[var] == c).mean()
            p_admin_ipw = np.average(admin[var] == c, weights=ipw_weights)
            p_jvs = np.average(jvs[var] == c, weights=jvs["weight"])
            result.append({"Kategoria": c,
                           "CBOP (raw)": round(p_admin_raw, 4),
                           "CBOP (IPW)": round(p_admin_ipw, 4),
                           "JVS (ważone)": round(p_jvs, 4)})
        return pd.DataFrame(result)
    else:
        return None


## Przykład 1: IPW z jedną zmienną ----
## P(R_A = 1 | size)
print("\n=== Przykład 1: P(R_A = 1 | size) ===\n")
res1 = ipw_estimate(admin, jvs, ["size"])
print(f"Szacunek IPW (single_shift): {res1['mean']:.4f}")
print(f"Szacowana wielkość populacji: {res1['N_hat']:.0f}")

## współczynniki modelu
print("\nWspółczynniki propensity score:")
print(res1["model"].params.round(4))

## wagi
print("\nRozkład wag IPW:")
print(pd.Series(res1["ipw_weights"]).describe().round(4))

## balans
print("\nBalans zmiennej 'size':")
print(check_balance(admin, jvs, "size", res1["ipw_weights"]))

print("\nBalans zmiennej 'private':")
print(check_balance(admin, jvs, "private", res1["ipw_weights"]))


## Przykład 2: IPW ze wszystkimi zmiennymi ----
## P(R_A = 1 | size, nace, region, private)
print("\n=== Przykład 2: P(R_A = 1 | size, nace, region, private) ===\n")
res2 = ipw_estimate(admin, jvs, wspol)
print(f"Szacunek IPW (single_shift): {res2['mean']:.4f}")
print(f"Szacowana wielkość populacji: {res2['N_hat']:.0f}")

## porównanie
print("\nPorównanie modeli:")
print(pd.DataFrame({
    "Model": ["~size", "~size+nace+region+private"],
    "Szacunek": [round(res1["mean"], 4), round(res2["mean"], 4)],
    "N_hat": [round(res1["N_hat"], 0), round(res2["N_hat"], 0)]
}))

## współczynniki modelu 2
print("\nWspółczynniki propensity score (model 2):")
print(res2["model"].params.round(4))

## histogram wag
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(res2["ipw_weights"], bins="fd", color="steelblue", edgecolor="white")
ax.set_xlabel("Waga IPW")
ax.set_ylabel("Liczba firm")
ax.set_title("Rozkład wag IPW (model 2)")
plt.tight_layout()
plt.show()

## balans
print("\nBalans zmiennej 'size' (model 1 vs model 2):")
print("Model 1:")
print(check_balance(admin, jvs, "size", res1["ipw_weights"]))
print("\nModel 2:")
print(check_balance(admin, jvs, "size", res2["ipw_weights"]))


## Porównanie rozkładów: wykres Love'a ----
def love_plot(admin, jvs, ipw_weights, formula_vars, title=""):
    """Wykres Love'a: |SMD| przed i po ważeniu IPW."""
    admin_dum = pd.get_dummies(admin[wspol], columns=["size", "nace", "region"], dtype=float)
    jvs_dum = pd.get_dummies(jvs[wspol], columns=["size", "nace", "region"], dtype=float)
    all_cols = sorted(set(admin_dum.columns) | set(jvs_dum.columns))
    admin_dum = admin_dum.reindex(columns=all_cols, fill_value=0)
    jvs_dum = jvs_dum.reindex(columns=all_cols, fill_value=0)
    w_jvs = jvs["weight"].values

    smd_raw, smd_adj = {}, {}
    for col in all_cols:
        ## raw
        m_a = admin_dum[col].mean()
        v_a = admin_dum[col].var()
        m_j = np.average(jvs_dum[col], weights=w_jvs)
        v_j = np.average((jvs_dum[col] - m_j)**2, weights=w_jvs)
        pool = np.sqrt((v_a + v_j) / 2)
        smd_raw[col] = (m_a - m_j) / pool if pool > 0 else 0

        ## adjusted
        m_a_w = np.average(admin_dum[col], weights=ipw_weights)
        v_a_w = np.average((admin_dum[col] - m_a_w)**2, weights=ipw_weights)
        pool_w = np.sqrt((v_a_w + v_j) / 2)
        smd_adj[col] = (m_a_w - m_j) / pool_w if pool_w > 0 else 0

    balance = pd.DataFrame({"Unadjusted": smd_raw, "Adjusted": smd_adj})
    balance = balance.sort_values("Unadjusted", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(balance) * 0.22)))
    y_pos = np.arange(len(balance))
    ax.scatter(balance["Unadjusted"].abs(), y_pos, c="coral", s=40, zorder=3, label="Unadjusted")
    ax.scatter(balance["Adjusted"].abs(), y_pos, c="steelblue", s=40, zorder=3, label="Adjusted")
    ax.axvline(0.1, ls="--", color="black", lw=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance.index, fontsize=6)
    ax.set_xlabel("|SMD|")
    ax.set_title(f"Balans zmiennych {title}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

print("\n--- Wykres Love'a: Model 1 ---")
love_plot(admin, jvs, res1["ipw_weights"], ["size"], title="(model 1: ~size)")

print("\n--- Wykres Love'a: Model 2 ---")
love_plot(admin, jvs, res2["ipw_weights"], wspol, title="(model 2: ~size+nace+region+private)")
