## ----
## Kalibrowany estymator IPW
## Autor: Maciej Beręsewicz
## ----

## Pakiety ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import fsolve
from scipy.special import expit

## Dane ----
jvs = pd.read_csv("../data/jvs.csv")
admin = pd.read_csv("../data/admin.csv")

print("admin (próba nielosowa S_A):")
print(admin.head())
print(f"\njvs (próba losowa S_B):")
print(jvs.head())

## Zmienne wspólne ----
wspol = ["private", "size", "nace", "region"]


## Funkcje pomocnicze ----

def fit_ps_mle(admin, jvs, formula_vars):
    """Estymacja PS metodą MLE (pseudo-likelihood z wagami JVS)."""
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
    ps_admin = ps_hat[y == 1].values
    return model, ps_admin, X.columns.tolist()


def fit_ps_gee(admin, jvs, formula_vars):
    """Estymacja PS metodą GEE (kalibracja do wartości globalnych z JVS).

    Rozwiązuje równanie:
    G(gamma) = sum_{S_A} x_i/pi(x_i,gamma) - sum_{S_B} d_i^B x_i = 0
    """
    ## przygotowanie macierzy
    admin_dum = pd.get_dummies(admin[formula_vars],
                                columns=[v for v in formula_vars if v != "private"],
                                dtype=float, drop_first=True)
    jvs_dum = pd.get_dummies(jvs[formula_vars],
                              columns=[v for v in formula_vars if v != "private"],
                              dtype=float, drop_first=True)
    all_cols = sorted(set(admin_dum.columns) | set(jvs_dum.columns))
    admin_dum = admin_dum.reindex(columns=all_cols, fill_value=0)
    jvs_dum = jvs_dum.reindex(columns=all_cols, fill_value=0)

    X_admin = sm.add_constant(admin_dum).values
    X_jvs = sm.add_constant(jvs_dum).values
    w_jvs = jvs["weight"].values

    ## sumy populacyjne (estymowane z JVS)
    tau_x = (X_jvs * w_jvs[:, None]).sum(axis=0)

    ## MLE jako wartość początkowa
    model_mle, _, _ = fit_ps_mle(admin, jvs, formula_vars)
    gamma0 = model_mle.params.values

    ## dopasowanie wymiarów (GEE może mieć inne kolumny niż MLE)
    col_names = ["const"] + all_cols
    if len(gamma0) != X_admin.shape[1]:
        gamma0 = np.zeros(X_admin.shape[1])

    def gee_equations(gamma):
        pi_admin = np.clip(expit(X_admin @ gamma), 1e-10, 1 - 1e-10)
        G = (X_admin / pi_admin[:, None]).sum(axis=0) - tau_x
        return G

    result = fsolve(gee_equations, gamma0, full_output=True)
    gamma_hat = result[0]

    ps_admin = 1 / (1 + np.exp(-X_admin @ gamma_hat))

    return gamma_hat, ps_admin, col_names


def fit_ps_gee_totals(admin, pop_totals, formula_vars):
    """GEE z wartościami globalnymi zamiast danych jednostkowych JVS.

    pop_totals: dict z nazwami zmiennych i ich sumami populacyjnymi.
    """
    admin_dum = pd.get_dummies(admin[formula_vars],
                                columns=[v for v in formula_vars if v != "private"],
                                dtype=float, drop_first=True)
    X_admin = sm.add_constant(admin_dum).values
    col_names = ["const"] + sorted(admin_dum.columns.tolist())

    ## wektor sum populacyjnych
    tau_x = np.array([pop_totals.get(c, 0) for c in col_names])

    gamma0 = np.zeros(X_admin.shape[1])

    def gee_equations(gamma):
        pi_admin = np.clip(expit(X_admin @ gamma), 1e-10, 1 - 1e-10)
        G = (X_admin / pi_admin[:, None]).sum(axis=0) - tau_x
        return G

    result = fsolve(gee_equations, gamma0, full_output=True)
    gamma_hat = result[0]
    ps_admin = 1 / (1 + np.exp(-X_admin @ gamma_hat))

    return gamma_hat, ps_admin, col_names


def ipw_mean(y, ps):
    """Estymator IPW Hajeka."""
    w = 1.0 / ps
    return np.average(y, weights=w)


def check_balance_df(admin, jvs, var, ipw_weights):
    """Porównanie proporcji: raw CBOP, IPW CBOP, JVS (ważone)."""
    cats = sorted(admin[var].unique())
    rows = []
    for c in cats:
        rows.append({
            "Kategoria": c,
            "CBOP (raw)": round((admin[var] == c).mean(), 4),
            "CBOP (IPW)": round(np.average(admin[var] == c, weights=ipw_weights), 4),
            "JVS (ważone)": round(np.average(jvs[var] == c, weights=jvs["weight"]), 4)
        })
    return pd.DataFrame(rows)


## ======================================================================
## Przykład 1: MLE vs GEE z jedną zmienną
## P(R_A = 1 | size)
## ======================================================================
print("\n=== Przykład 1: P(R_A = 1 | size) ===\n")

## MLE
model_mle, ps_mle, _ = fit_ps_mle(admin, jvs, ["size"])
w_mle = 1.0 / ps_mle
mu_mle = ipw_mean(admin["single_shift"], ps_mle)
print(f"IPW-MLE (single_shift): {mu_mle:.4f}")

## GEE
gamma_gee, ps_gee, col_gee = fit_ps_gee(admin, jvs, ["size"])
w_gee = 1.0 / ps_gee
mu_gee = ipw_mean(admin["single_shift"], ps_gee)
print(f"IPW-GEE (single_shift): {mu_gee:.4f}")

## balans
print("\nBalans 'size' -- MLE:")
print(check_balance_df(admin, jvs, "size", w_mle))
print("\nBalans 'size' -- GEE:")
print(check_balance_df(admin, jvs, "size", w_gee))

## współczynniki
print(f"\nWspółczynniki MLE:\n{model_mle.params.round(4)}")
print(f"\nWspółczynniki GEE:\n{pd.Series(gamma_gee, index=col_gee).round(4)}")


## ======================================================================
## Przykład 2: MLE vs GEE ze wszystkimi zmiennymi
## P(R_A = 1 | size, nace, region, private)
## ======================================================================
print("\n=== Przykład 2: P(R_A = 1 | size, nace, region, private) ===\n")

## MLE
model_mle2, ps_mle2, _ = fit_ps_mle(admin, jvs, wspol)
w_mle2 = 1.0 / ps_mle2
mu_mle2 = ipw_mean(admin["single_shift"], ps_mle2)
print(f"IPW-MLE (single_shift): {mu_mle2:.4f}")

## GEE
gamma_gee2, ps_gee2, col_gee2 = fit_ps_gee(admin, jvs, wspol)
w_gee2 = 1.0 / ps_gee2
mu_gee2 = ipw_mean(admin["single_shift"], ps_gee2)
print(f"IPW-GEE (single_shift): {mu_gee2:.4f}")

## porównanie
print("\nPorównanie oszacowań:")
print(pd.DataFrame({
    "Metoda": ["MLE", "GEE"],
    "Szacunek": [round(mu_mle2, 4), round(mu_gee2, 4)],
    "N_hat": [round(w_mle2.sum(), 0), round(w_gee2.sum(), 0)]
}))

## porównanie wag
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(w_mle2, w_gee2, alpha=0.3, s=10)
lims = [0, max(w_mle2.max(), w_gee2.max()) * 1.05]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("Wagi IPW (MLE)")
ax.set_ylabel("Wagi IPW (GEE)")
ax.set_title("Porównanie wag MLE vs GEE")
plt.tight_layout()
plt.show()

## rozkład wag
print(f"\nWagi MLE:  {pd.Series(w_mle2).describe().round(2).to_dict()}")
print(f"Wagi GEE:  {pd.Series(w_gee2).describe().round(2).to_dict()}")

## balans
print("\nBalans 'size' -- MLE:")
print(check_balance_df(admin, jvs, "size", w_mle2))
print("\nBalans 'size' -- GEE:")
print(check_balance_df(admin, jvs, "size", w_gee2))


## ======================================================================
## Przykład 3: GEE z wartościami globalnymi (bez danych jednostkowych JVS)
## ======================================================================
print("\n=== Przykład 3: GEE z wartościami globalnymi ===\n")

## sumy populacyjne (znane)
pop_totals = {"const": 51870, "size_M": 13758, "size_S": 29551}

gamma_gee3, ps_gee3, col_gee3 = fit_ps_gee_totals(admin, pop_totals, ["size"])
w_gee3 = 1.0 / ps_gee3
mu_gee3 = ipw_mean(admin["single_shift"], ps_gee3)

print(f"IPW-GEE z danymi jednostkowymi:   {mu_gee:.4f}")
print(f"IPW-GEE z wartościami globalnymi: {mu_gee3:.4f}")
