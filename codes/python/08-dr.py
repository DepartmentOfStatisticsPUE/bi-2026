## ----
## Estymator podwójnie odporny (DR)
## Autor: Maciej Beręsewicz
##
## Estymator DR:
##   mu_DR = (1 / N_hat_B) * { sum_{S_A} (y_i - m_hat_i) / pi_hat_i
##                            + sum_{S_B} d_i^B * m_hat_i }
##
## gdzie:
##   pi_hat_i  = pi(x_i, gamma_hat)  -- model selekcji (PS)
##   m_hat_i   = m(x_i, beta_hat)    -- model wynikowy (GLM)
##   N_hat_B   = sum_{S_B} d_i^B      -- estymator wielkości populacji
##
## Estymator jest asymptotycznie nieobciążony, gdy przynajmniej jeden
## z dwóch modeli (PS lub wynikowy) jest poprawnie wyspecyfikowany.
## ----

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import fsolve

## Dane ----
admin = pd.read_csv("../data/admin.csv")
jvs = pd.read_csv("../data/jvs.csv")

## region jako tekst zero-padded (zgodnie z R)
admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"] = jvs["region"].astype(str).str.zfill(2)

w_jvs = jvs["weight"].values
N_pop = w_jvs.sum()


## Funkcje pomocnicze ----
def design_X(admin, jvs, formula_vars):
    """Wspólna macierz X (z constem) dla obu prób — dummy encoding
    dla wszystkich poza 'private'."""
    cat_vars = [v for v in formula_vars if v != "private"]
    a = pd.get_dummies(admin[formula_vars], columns=cat_vars,
                       dtype=float, drop_first=True)
    j = pd.get_dummies(jvs[formula_vars], columns=cat_vars,
                       dtype=float, drop_first=True)
    cols = sorted(set(a.columns) | set(j.columns))
    a = a.reindex(columns=cols, fill_value=0)
    j = j.reindex(columns=cols, fill_value=0)
    X_admin = sm.add_constant(a, has_constant="add").values.astype(float)
    X_jvs = sm.add_constant(j, has_constant="add").values.astype(float)
    return X_admin, X_jvs


def fit_ps_mle(X_admin, X_jvs, w_jvs):
    """Pseudo-MLE PS (Chen, Li, Wu 2020):
       sum_{S_A} x_i - sum_{S_B} d_i^B * pi_i * x_i = 0."""
    def score(gamma):
        pi_jvs = 1.0 / (1.0 + np.exp(-(X_jvs @ gamma)))
        return X_admin.sum(axis=0) - (w_jvs * pi_jvs) @ X_jvs
    g0 = np.zeros(X_admin.shape[1])
    return fsolve(score, g0)


def fit_ps_gee(X_admin, X_jvs, w_jvs):
    """GEE / kalibracyjny IPW: dopasowanie tak, by
       sum_{S_A} x_i / pi_i = sum_{S_B} d_i^B * x_i."""
    T = (w_jvs[:, None] * X_jvs).sum(axis=0)
    def score(gamma):
        pi_admin = 1.0 / (1.0 + np.exp(-(X_admin @ gamma)))
        return (X_admin / pi_admin[:, None]).sum(axis=0) - T
    g0 = np.zeros(X_admin.shape[1])
    return fsolve(score, g0)


def fit_outcome(X_admin, y_admin, family="binomial"):
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    return sm.GLM(y_admin, X_admin, family=fam).fit()


def dr_estimate(admin, jvs, sel_vars, out_vars, target,
                family_outcome="binomial", ps_method="mle"):
    """Oszacowanie estymatora DR."""
    Xs_admin, Xs_jvs = design_X(admin, jvs, sel_vars)
    Xo_admin, Xo_jvs = design_X(admin, jvs, out_vars)
    y_admin = admin[target].values
    w = jvs["weight"].values

    fit_ps = fit_ps_mle if ps_method == "mle" else fit_ps_gee
    gamma = fit_ps(Xs_admin, Xs_jvs, w)
    pi_admin = 1.0 / (1.0 + np.exp(-(Xs_admin @ gamma)))

    out_model = fit_outcome(Xo_admin, y_admin, family=family_outcome)
    m_admin = out_model.predict(Xo_admin)
    m_jvs = out_model.predict(Xo_jvs)

    N_hat_B = w.sum()
    mu_dr = (np.sum((y_admin - m_admin) / pi_admin)
             + np.sum(w * m_jvs)) / N_hat_B
    return {"mean": mu_dr, "pi_admin": pi_admin,
            "m_admin": m_admin, "m_jvs": m_jvs}


## Przykład 1: DR z modelem liniowym (gaussian) ----
##              ten sam zestaw zmiennych w PS i modelu wynikowym
res_gauss = dr_estimate(admin, jvs, ["size", "nace"], ["size", "nace"],
                        "single_shift", family_outcome="gaussian")
print(f"DR glm gauss   (size+nace):       {res_gauss['mean']:.4f}")


## Przykład 2: DR z modelem logistycznym (binomial) ----
res_binom = dr_estimate(admin, jvs, ["size", "nace"], ["size", "nace"],
                        "single_shift", family_outcome="binomial")
print(f"DR glm binom   (size+nace):       {res_binom['mean']:.4f}")


## Przykład 3: DR z PS estymowanym metodą GEE ----
res_gee = dr_estimate(admin, jvs, ["size", "nace"], ["size", "nace"],
                      "single_shift", family_outcome="binomial",
                      ps_method="gee")
print(f"DR gee binom   (size+nace):       {res_gee['mean']:.4f}")


## Przykład 4: DR z pełnym zestawem zmiennych w obu modelach ----
wspol = ["size", "nace", "region", "private"]
res_full = dr_estimate(admin, jvs, wspol, wspol,
                       "single_shift", family_outcome="binomial")
print(f"DR glm binom   (pełny):           {res_full['mean']:.4f}")


## Porównanie ----
print("\n--- Porównanie wariantów DR ---")
print(pd.DataFrame({
    "Wariant": ["DR glm gauss   (size+nace)",
                "DR glm binom   (size+nace)",
                "DR gee binom   (size+nace)",
                "DR glm binom   (pełny)"],
    "Oszacowanie": [round(res_gauss["mean"], 4),
                    round(res_binom["mean"], 4),
                    round(res_gee["mean"], 4),
                    round(res_full["mean"], 4)]
}))


## Porównanie z IPW i MI-GLM (pełny model) ----
## IPW Horvitz-Thompson (por. CLAUDE.md): mu_IPW = sum_{S_A} y / pi / N_pop
def ipw_estimate(admin, jvs, sel_vars, target):
    Xs_admin, Xs_jvs = design_X(admin, jvs, sel_vars)
    w = jvs["weight"].values
    gamma = fit_ps_mle(Xs_admin, Xs_jvs, w)
    pi_admin = 1.0 / (1.0 + np.exp(-(Xs_admin @ gamma)))
    return np.sum(admin[target].values / pi_admin) / w.sum()


def mi_estimate(admin, jvs, out_vars, target, family="binomial"):
    Xo_admin, Xo_jvs = design_X(admin, jvs, out_vars)
    out_model = fit_outcome(Xo_admin, admin[target].values, family=family)
    m_jvs = out_model.predict(Xo_jvs)
    w = jvs["weight"].values
    return np.sum(w * m_jvs) / w.sum()


mu_ipw = ipw_estimate(admin, jvs, wspol, "single_shift")
mu_mi = mi_estimate(admin, jvs, wspol, "single_shift", family="binomial")

print("\n--- IPW vs MI-GLM vs DR (pełny model) ---")
print(pd.DataFrame({
    "Metoda": ["IPW (pełny)",
               "MI-GLM binom (pełny)",
               "DR  binom    (pełny)"],
    "Oszacowanie": [round(mu_ipw, 4),
                    round(mu_mi, 4),
                    round(res_full["mean"], 4)]
}))


## ======================================================================
## Zadanie 1: różne specyfikacje modelu wynikowego
## ======================================================================
## Zafiksuj model selekcji jako [size, nace, region, private] i dopasuj
## cztery estymatory DR z rosnącą specyfikacją modelu wynikowego
## (wszystkie z family_outcome="binomial"):
##   - DR-A: outcome = [size]
##   - DR-B: outcome = [size, nace]
##   - DR-C: outcome = [size, nace, region]
##   - DR-D: outcome = [size, nace, region, private]
##
## Zestawienie wyniki w jednej tabeli i odpowiedz:
## a) Jak zmienia się mean wraz z dodawaniem zmiennych do modelu wynikowego?
## b) Czy oszacowania zbiegają do jednej wartości? Co to mówi o roli
##    modelu wynikowego, gdy PS jest dobrze wyspecyfikowany?
## c) Porównaj DR-D z czystym IPW (zmienne pełne) -- w którym przypadku
##    oszacowania są bliższe?

## ======================================================================
## Zadanie 2: różne specyfikacje modelu selekcji
## ======================================================================
## Zafiksuj model wynikowy jako [size, nace, region, private] (binomial)
## i zmieniaj tylko model selekcji:
##   - DR-S1: selection = [size]
##   - DR-S2: selection = [size, nace]
##   - DR-S3: selection = [size, nace, region, private]
##
## a) Jak zmienia się mean? Czy efekt jest mniejszy niż w Zadaniu 1?
## b) Sformułuj wniosek: który z dwóch modeli (PS czy wynikowy) ma
##    większy wpływ na oszacowanie DR w tym przykładzie?

## ======================================================================
## Zadanie 3: celowe zepsucie jednego z modeli
## ======================================================================
## Sprawdź własność podwójnej odporności empirycznie. Wykorzystaj fakt,
## że single_shift jest najsilniej powiązany ze zmiennymi size i nace.
##
## a) Zły PS, dobry model wynikowy:
##    selection = [private],
##    outcome = [size, nace, region, private] (binomial).
## b) Dobry PS, zły model wynikowy:
##    selection = [size, nace, region, private],
##    outcome = [private] (binomial).
## c) Oba złe:
##    selection = [private], outcome = [private] (binomial).
##
## Zestawienie oszacowania mean w tabeli i porównaj z DR-D z Zadania 1
## (oba modele dobre). Czy w wariantach (a) i (b) oszacowania są zbliżone
## do (d)? Co dzieje się w wariancie (c)?

## ======================================================================
## Zadanie 4: GEE vs MLE w roli PS
## ======================================================================
## Dla zestawu zmiennych [size, nace, region, private] dopasuj dwa
## estymatory DR (oba z family_outcome="binomial"):
##   - DR-MLE: ps_method="mle".
##   - DR-GEE: ps_method="gee".
##
## a) Porównaj oszacowania mean.
## b) Zbuduj rozkład wag IPW (1 / pi_admin) dla obu wariantów -- który
##    daje mniejszą rozpiętość wag?
## c) Sprawdź balans zmiennej size: ważone proporcje w S_A (z wagami
##    1/pi) vs w S_B (z wagami d^B). Który PS daje lepszą zgodność?
