## ----
## Masowa imputacja: model GLM (Kim et al. 2021)
## Autor: Maciej Beręsewicz
##
## Estymator MI-K:  mu_MI = sum_{S_B} d_i^B * y_hat_i / sum_{S_B} d_i^B
##
## Procedura:
##   1. Dopasuj model na próbie nielosowej S_A
##   2. Predykuj y_hat na próbie losowej S_B
##   3. Oblicz ważoną średnią z wagami d^B
## ----

import pandas as pd
import numpy as np
import statsmodels.api as sm

## Dane ----
admin = pd.read_csv("../data/admin.csv")
jvs = pd.read_csv("../data/jvs.csv")

## region jako tekst zero-padded (zgodnie z R)
admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"] = jvs["region"].astype(str).str.zfill(2)

w_jvs = jvs["weight"].values


## Funkcja pomocnicza: dopasowanie GLM i predykcja MI ----
def mi_glm(admin, jvs, formula_vars, cat_vars, target, family="binomial"):
    """Mass imputation z modelem GLM (Kim et al. 2021)."""
    admin_dum = pd.get_dummies(admin[formula_vars], columns=cat_vars,
                               dtype=float, drop_first=True)
    jvs_dum = pd.get_dummies(jvs[formula_vars], columns=cat_vars,
                             dtype=float, drop_first=True)
    all_cols = sorted(set(admin_dum.columns) | set(jvs_dum.columns))
    admin_dum = admin_dum.reindex(columns=all_cols, fill_value=0)
    jvs_dum = jvs_dum.reindex(columns=all_cols, fill_value=0)

    X_admin = sm.add_constant(admin_dum).values.astype(float)
    X_jvs = sm.add_constant(jvs_dum).values.astype(float)
    y_admin = admin[target].values

    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    model = sm.GLM(y_admin, X_admin, family=fam).fit()
    y_hat_jvs = model.predict(X_jvs)

    return model, y_hat_jvs


## Przykład 1: model regresji liniowej (gaussian) ~size
model, y_hat = mi_glm(admin, jvs, ["size"], ["size"],
                      "single_shift", family="gaussian")
mu_glm1 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-GLM (gauss, ~size):            {mu_glm1:.4f}")


## Przykład 2: model regresji logistycznej (binomial) ~size
model, y_hat = mi_glm(admin, jvs, ["size"], ["size"],
                      "single_shift", family="binomial")
mu_glm2 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-GLM (binom, ~size):            {mu_glm2:.4f}")


## Przykład 3: pełny model logistyczny
model, y_hat = mi_glm(admin, jvs,
                      ["size", "nace", "region", "private"],
                      ["size", "nace", "region"],
                      "single_shift", family="binomial")
mu_glm3 = np.sum(w_jvs * y_hat) / np.sum(w_jvs)
print(f"MI-GLM (binom, pełny):            {mu_glm3:.4f}")

print("\nWspółczynniki modelu pełnego:")
print(pd.Series(model.params, index=["const"] + sorted(
    pd.get_dummies(admin[["size", "nace", "region", "private"]],
                   columns=["size", "nace", "region"],
                   dtype=float, drop_first=True).columns.tolist())).round(4))


## ======================================================================
## Porównanie
## ======================================================================
print("\n--- Porównanie ---")
print(pd.DataFrame({
    "Metoda": ["MI-GLM gauss  (~size)",
               "MI-GLM binom  (~size)",
               "MI-GLM binom  (pełny)"],
    "Oszacowanie": [round(mu_glm1, 4), round(mu_glm2, 4), round(mu_glm3, 4)]
}))


## Ćwiczenie:
## - Porównaj wyniki MI-GLM (binomial) z MI-NN i MI-PMM z 06-mi-nn.py
## - Dodaj kolejne zmienne pomocnicze i sprawdź jak zmienia się oszacowanie
## - Porównaj estymator MI-GLM z estymatorami IPW (03-ipw-1.py)
