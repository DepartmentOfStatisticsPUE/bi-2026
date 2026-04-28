## ----
## Masowa imputacja: najbliższy sąsiad (NN) i predictive mean matching (PMM)
## Autor: Maciej Beręsewicz
##
## Estymator MI:  mu_MI = sum_{S_B} d_i^B * y_hat_i / sum_{S_B} d_i^B
## ----

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

## Dane ----
admin = pd.read_csv("../data/admin.csv")
jvs = pd.read_csv("../data/jvs.csv")

## region jako tekst zero-padded (zgodnie z R)
admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"] = jvs["region"].astype(str).str.zfill(2)

w_jvs = jvs["weight"].values
N_pop = w_jvs.sum()


## Funkcja pomocnicza: design matrix dla cech ----
def make_design(df, formula_vars, cat_vars, all_cols=None):
    """Tworzy macierz X z dummy-encoding zgodnym z R (factor)."""
    dum = pd.get_dummies(df[formula_vars], columns=cat_vars,
                         dtype=float, drop_first=True)
    if all_cols is not None:
        dum = dum.reindex(columns=all_cols, fill_value=0)
    return dum


## ======================================================================
## Podejście I: nearest neighbor matching (NN)
## ======================================================================

def mi_nn(admin, jvs, formula_vars, cat_vars, target, k=5):
    """MI-NN: dla każdej obserwacji jvs znajdź k najbliższych w admin
    w przestrzeni X i przypisz średnie y."""
    X_admin = make_design(admin, formula_vars, cat_vars).values
    X_jvs = make_design(jvs, formula_vars, cat_vars,
                        all_cols=make_design(admin, formula_vars, cat_vars).columns).values
    y_admin = admin[target].values

    nn = NearestNeighbors(n_neighbors=k).fit(X_admin)
    _, idx = nn.kneighbors(X_jvs)
    y_hat_jvs = y_admin[idx].mean(axis=1)
    return y_hat_jvs


## Przykład 1: NN, ~size, k=5
y_hat = mi_nn(admin, jvs, ["size"], ["size"], "single_shift", k=5)
mu_nn1 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-NN  (~size, k=5):              {mu_nn1:.4f}")


## Przykład 2: NN, pełny model, k=5
y_hat = mi_nn(admin, jvs,
              ["size", "nace", "region", "private"],
              ["size", "nace", "region"],
              "single_shift", k=5)
mu_nn2 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-NN  (pełny, k=5):              {mu_nn2:.4f}")


## ======================================================================
## Podejście II: predictive mean matching (PMM)
## ======================================================================

def mi_pmm(admin, jvs, formula_vars, cat_vars, target,
           family="binomial", k=5):
    """MI-PMM: dopasuj model na admin, predykuj y_hat na admin i jvs,
    znajdź k najbliższych obserwacji admin do każdej jvs po y_hat."""
    X_admin_dum = make_design(admin, formula_vars, cat_vars)
    X_jvs_dum = make_design(jvs, formula_vars, cat_vars,
                            all_cols=X_admin_dum.columns)
    X_admin = sm.add_constant(X_admin_dum).values
    X_jvs = sm.add_constant(X_jvs_dum).values
    y_admin = admin[target].values

    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    model = sm.GLM(y_admin, X_admin, family=fam).fit()
    yhat_admin = model.predict(X_admin)
    yhat_jvs = model.predict(X_jvs)

    nn = NearestNeighbors(n_neighbors=k).fit(yhat_admin.reshape(-1, 1))
    _, idx = nn.kneighbors(yhat_jvs.reshape(-1, 1))
    y_hat_final = y_admin[idx].mean(axis=1)
    return y_hat_final


## Przykład 3: PMM, ~size, binomial
y_hat = mi_pmm(admin, jvs, ["size"], ["size"], "single_shift",
               family="binomial", k=5)
mu_pmm1 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-PMM (~size binom, k=5):        {mu_pmm1:.4f}")


## Przykład 4: PMM, pełny model, binomial
y_hat = mi_pmm(admin, jvs,
               ["size", "nace", "region", "private"],
               ["size", "nace", "region"],
               "single_shift", family="binomial", k=5)
mu_pmm2 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-PMM (pełny binom, k=5):        {mu_pmm2:.4f}")


## ======================================================================
## Porównanie
## ======================================================================
print("\n--- Porównanie ---")
print(pd.DataFrame({
    "Metoda": ["MI-NN  (~size)", "MI-NN  (pełny)",
               "MI-PMM (~size)", "MI-PMM (pełny, binomial)"],
    "Oszacowanie": [round(mu_nn1, 4), round(mu_nn2, 4),
                    round(mu_pmm1, 4), round(mu_pmm2, 4)]
}))


## Ćwiczenie:
## - Sprawdź jak zmienia się oszacowanie dla różnych wartości k (1, 3, 5, 10)
## - Porównaj wyniki MI-NN i MI-PMM dla tego samego zestawu zmiennych
## - Sprawdź jak wpływa rodzina (gaussian vs binomial) na wynik PMM
