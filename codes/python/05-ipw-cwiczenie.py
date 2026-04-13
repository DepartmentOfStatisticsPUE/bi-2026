## ----
## Cwiczenie: IPW krok po kroku (MLE i kalibrowany)
## Autor: Maciej Beresewicz
##
## UWAGA: w tym cwiczeniu stosujemy estymator IPW 1 (Horvitza-Thompsona),
## w ktorym mianownikiem jest N_pop = sum(d_i^B).
## Tak domyslnie dziala funkcja nonprob() w R.
## Estymator IPW 2 (Hajeka) z mianownikiem sum(w_i) omowimy osobno.
## ----

## =============================================
## Czesc A: Dane i estymator naiwny
## =============================================

## Pakiety ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import fsolve
from scipy.special import expit

## Dane ----
admin = pd.read_csv("../data/admin.csv")
jvs = pd.read_csv("../data/jvs.csv")

## region jako tekst z wiodacym zerem (zgodnosc z R)
admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"] = jvs["region"].astype(str).str.zfill(2)

## wielkosc populacji estymowana z JVS
N_pop = jvs["weight"].sum()

print("admin (proba nielosowa S_A):")
print(admin.head())
print(f"\njvs (proba losowa S_B):")
print(jvs.head())

## Zadanie A1: Porownaj rozklad zmiennej size w obu zrodlach
## Tutaj kod


## Zadanie A2: Oblicz naiwna srednia single_shift z admin (bez korekt).
## Zapisz wynik -- porownasz go z estymatorami IPW w kolejnych czesciach.
## Tutaj kod


## =============================================
## Czesc B: Estymator IPW-MLE
## =============================================

## Przyklad: model z jedna zmienna (size) ----
## Pseudo-MLE: rozwiazujemy rownanie score
## U(gamma) = sum_{S_A} x_i - sum_{S_B} d_i^B * pi(x_i, gamma) * x_i = 0

formula_vars_simple = ["size"]
cat_vars_simple = ["size"]

admin_dum_s = pd.get_dummies(admin[formula_vars_simple], columns=cat_vars_simple,
                              dtype=float, drop_first=True)
jvs_dum_s = pd.get_dummies(jvs[formula_vars_simple], columns=cat_vars_simple,
                            dtype=float, drop_first=True)
all_cols_s = sorted(set(admin_dum_s.columns) | set(jvs_dum_s.columns))
admin_dum_s = admin_dum_s.reindex(columns=all_cols_s, fill_value=0)
jvs_dum_s = jvs_dum_s.reindex(columns=all_cols_s, fill_value=0)

X_admin_s = sm.add_constant(admin_dum_s).values.astype(float)
X_jvs_s = sm.add_constant(jvs_dum_s).values.astype(float)
w_jvs_s = jvs["weight"].values

def score_eq_simple(gamma):
    pi_jvs = expit(X_jvs_s @ gamma)
    return X_admin_s.sum(axis=0) - (X_jvs_s * (w_jvs_s * pi_jvs)[:, None]).sum(axis=0)

gamma_hat_simple = fsolve(score_eq_simple, np.zeros(X_admin_s.shape[1]))
ps_simple = expit(X_admin_s @ gamma_hat_simple)
w_simple = 1.0 / ps_simple

## estymator HT
mu_simple = np.sum(w_simple * admin["single_shift"].values) / N_pop
print(f"\nOszacowanie IPW-MLE (~size): {mu_simple:.4f}")
print(f"Rozklad wag:")
print(pd.Series(w_simple).describe().round(2))


## Funkcja do sprawdzania balansu ----
def check_balance(admin, jvs, var, ipw_weights):
    cats = sorted(admin[var].unique())
    rows = []
    for c in cats:
        rows.append({
            "Kategoria": c,
            "CBOP (raw)": round((admin[var] == c).mean(), 4),
            "CBOP (IPW)": round(np.average(admin[var] == c, weights=ipw_weights), 4),
            "JVS (wazone)": round(np.average(jvs[var] == c, weights=jvs["weight"]), 4)
        })
    return pd.DataFrame(rows)


## Zadanie B1: Zmodyfikuj powyzszy kod tak, aby model wykorzystywal
## WSZYSTKIE zmienne: private, size, nace, region
## Podpowiedz: zmien formula_vars i cat_vars
## formula_vars = ["private", "size", "nace", "region"]
## cat_vars = ["size", "nace", "region"]
## Tutaj kod


## Zadanie B2: Dla modelu pelnego:
## a) Wypisz rozklad wag (pd.Series(w).describe())
## b) Sprawdz balans zmiennej size: check_balance(admin, jvs, "size", twoje_wagi)
## Tutaj kod


## =============================================
## Czesc C: Kalibrowany estymator IPW (GEE)
## =============================================

## Zadanie C1: Napisz model GEE z wszystkimi zmiennymi
## Ponizej szablon -- uzupelnij formula_vars
## formula_vars = ["private", "size", "nace", "region"]  ## <-- odkomentuj i uzupelnij
## cat_vars = ["size", "nace", "region"]

## admin_dum = pd.get_dummies(admin[formula_vars],
##                             columns=cat_vars,
##                             dtype=float, drop_first=True)
## jvs_dum = pd.get_dummies(jvs[formula_vars],
##                           columns=cat_vars,
##                           dtype=float, drop_first=True)
## all_cols = sorted(set(admin_dum.columns) | set(jvs_dum.columns))
## admin_dum = admin_dum.reindex(columns=all_cols, fill_value=0)
## jvs_dum = jvs_dum.reindex(columns=all_cols, fill_value=0)
##
## X_admin = sm.add_constant(admin_dum).values
## X_jvs = sm.add_constant(jvs_dum).values
## w_jvs = jvs["weight"].values
##
## tau_x = (X_jvs * w_jvs[:, None]).sum(axis=0)
##
## def gee_equations(gamma):
##     pi_admin = np.clip(expit(X_admin @ gamma), 1e-10, 1 - 1e-10)
##     return (X_admin / pi_admin[:, None]).sum(axis=0) - tau_x
##
## gamma0 = np.zeros(X_admin.shape[1])
## gamma_gee = fsolve(gee_equations, gamma0)
## ps_gee = expit(X_admin @ gamma_gee)
## w_gee = 1.0 / ps_gee
##
## mu_gee = np.sum(w_gee * admin["single_shift"].values) / N_pop
## print(f"Oszacowanie IPW-GEE: {mu_gee:.4f}")


## Zadanie C2: Porownaj MLE vs GEE
## a) Wypisz oszacowania obu metod
## b) Sprawdz balans zmiennej size dla obu metod
## c) Narysuj wykres: plt.scatter(w_mle, w_gee)
## Tutaj kod


## Zadanie C3: GEE z wartosciami globalnymi ----
pop_totals = {"const": 51870, "size_M": 13758, "size_S": 29551}

admin_dum_s = pd.get_dummies(admin[["size"]], columns=["size"],
                              dtype=float, drop_first=True)
X_admin_s = sm.add_constant(admin_dum_s).values
col_names_s = ["const"] + sorted(admin_dum_s.columns.tolist())
tau_x_s = np.array([pop_totals.get(c, 0) for c in col_names_s])

def gee_eq_totals(gamma):
    pi_a = np.clip(expit(X_admin_s @ gamma), 1e-10, 1 - 1e-10)
    return (X_admin_s / pi_a[:, None]).sum(axis=0) - tau_x_s

result_tot = fsolve(gee_eq_totals, np.zeros(X_admin_s.shape[1]))
ps_tot = expit(X_admin_s @ result_tot)
mu_gee_tot = np.sum((1.0 / ps_tot) * admin["single_shift"].values) / N_pop
print(f"IPW-GEE (wartosci globalne): {mu_gee_tot:.4f}")


## =============================================
## Czesc D: Podsumowanie
## =============================================

## Zadanie D1: Uzupelnij tabele swoimi wynikami
## Zadanie D2: Odpowiedz na pytania:
## a) Jak zmienily sie oszacowania po zastosowaniu IPW vs naiwny?
## b) Kiedy GEE lepszy od MLE?
## c) Kiedy uzywasz pop_totals?
