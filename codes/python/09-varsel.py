"""
Dobór zmiennych do modelu (LASSO / SCAD / MCP)
================================================

Wersja Python notatnika 09-varsel. Polski.

Pakiety:
- pycasso (https://github.com/jasonge27/picasso) -- LASSO / SCAD / MCP
  (instalacja: pip install pycasso). Nie obsługuje sample_weight, więc
  używamy go w Części A (regresja na sztucznych danych).
- sklearn LogisticRegression z penalty="l1" -- używamy w Częściach B i C
  do ważonej regresji (sample_weight) dla pseudo-IPW. Dla pełnej obsługi
  SCAD/MCP z wagami patrz wersja R (nonprobsvy).
"""

# %% Pakiety -----------------------------------------------------------------
import numpy as np
import pandas as pd
import pycasso
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(2026)


# %% ============================================================
# %% CZĘŚĆ A: Sztuczny przykład regresji liniowej -- LASSO/SCAD/MCP
# %% ============================================================

# Generujemy n=200, p=50, 5 aktywnych zmiennych:
# y = 3*x1 + 1.5*x2 + 2*x5 - 2.5*x10 + 1*x20 + eps,  eps ~ N(0,1)
n, p = 200, 50
X = rng.standard_normal((n, p))

beta_true = np.zeros(p)
beta_true[[0, 1, 4, 9, 19]] = [3.0, 1.5, 2.0, -2.5, 1.0]
y = X @ beta_true + rng.standard_normal(n)
which_active = np.where(beta_true != 0)[0]
print("Prawdziwie aktywne zmienne (0-indexed):", which_active)


# %% Dopasowanie pełnej ścieżki dla LASSO / SCAD / MCP -----------------------
# pycasso.Solver oblicza ścieżkę 100 wartości lambda; coef()['beta']
# zwraca macierz (nlambda x p) -- wiersz = jedna wartość lambda.

solvers = {}
for pen in ["l1", "scad", "mcp"]:
    s = pycasso.Solver(X, y, family="gaussian", penalty=pen)
    s.train()
    solvers[pen] = s


# %% Wybór lambda przez ręczną walidację krzyżową ----------------------------
# pycasso nie ma wbudowanego CV -- piszemy proste 10-fold CV
# wybierające lambda minimalizujące MSE na zbiorze walidacyjnym.

def cv_pycasso(X, y, penalty, k_folds=10, seed=2026, n_lambdas=50):
    """Prosta walidacja krzyżowa dla pycasso.

    Zwraca: (best_j, lambdas, beta_best) gdzie best_j to indeks lambda
    minimalizujący CV-MSE, beta_best -- wektor współczynników w optimum
    (z dopasowania na pełnych danych).
    """
    rng_cv = np.random.default_rng(seed)
    idx = rng_cv.permutation(len(y))
    folds = np.array_split(idx, k_folds)
    # explicit lambda sequence
    lam_max = np.max(np.abs(X.T @ (y - y.mean()))) / len(y)
    lambdas = np.exp(np.linspace(np.log(lam_max),
                                  np.log(0.001 * lam_max), n_lambdas))

    # dopasowanie na pełnych danych
    s_full = pycasso.Solver(X, y, family="gaussian", penalty=penalty,
                             lambdas=lambdas)
    s_full.train()
    beta_full = s_full.coef()["beta"]
    n_full = beta_full.shape[0]  # może być krótsza niż n_lambdas

    mse = np.zeros(n_full)
    counts = np.zeros(n_full)
    for f in folds:
        train_idx = np.setdiff1d(np.arange(len(y)), f)
        s_fold = pycasso.Solver(X[train_idx], y[train_idx],
                                 family="gaussian", penalty=penalty,
                                 lambdas=lambdas[:n_full])
        s_fold.train()
        bp = s_fold.coef()["beta"]
        ip = s_fold.coef()["intercept"]
        for j in range(min(bp.shape[0], n_full)):
            y_pred = X[f] @ bp[j] + ip[j]
            mse[j] += ((y[f] - y_pred) ** 2).sum()
            counts[j] += len(f)
    mse = np.where(counts > 0, mse / np.maximum(counts, 1), np.inf)
    best_j = int(np.argmin(mse))
    return best_j, lambdas[:n_full], beta_full[best_j]


results_a = []
for pen in ["l1", "scad", "mcp"]:
    best_j, lams, beta_best = cv_pycasso(X, y, penalty=pen)
    sel = np.where(np.abs(beta_best) > 1e-6)[0]
    results_a.append({
        "metoda": pen.upper().replace("L1", "LASSO"),
        "lambda_opt": round(lams[best_j], 4),
        "n_wybranych": len(sel),
        "prawdziwe_TP": int(np.isin(which_active, sel).sum()),
        "falszywe_FP": int((~np.isin(sel, which_active)).sum()),
        "sel": sel.tolist()
    })

print("\nPorównanie LASSO / SCAD / MCP (Część A):")
print(pd.DataFrame(results_a).to_string(index=False))

# Oczekiwane: LASSO wybiera więcej zmiennych (więcej FP);
# SCAD i MCP dzięki własności wyroczni są bardziej oszczędne.


# %% ============================================================
# %% CZĘŚĆ B: Sztuczny przykład z próbami losową i nielosową
# %% ============================================================

# Populacja N=10000, p=20 zmiennych
N, p = 10000, 20
X_pop = rng.standard_normal((N, p))

# y zależy tylko od x1, x2
y_pop = 1 + 2 * X_pop[:, 0] - 1.5 * X_pop[:, 1] + rng.standard_normal(N)
mu_true = y_pop.mean()

# prawdopodobieństwo włączenia do S_A zależy od x3, x4
eta = -2 + X_pop[:, 2] - 0.5 * X_pop[:, 3]
pi_A = 1 / (1 + np.exp(-eta))
R_A = rng.binomial(1, pi_A)

# próba probabilistyczna S_B: SRS o liczebności n_B=500
n_B = 500
idx_B = rng.choice(N, size=n_B, replace=False)

X_A = X_pop[R_A == 1]
y_A = y_pop[R_A == 1]
n_A = X_A.shape[0]

X_B = X_pop[idx_B]
d_B = np.full(n_B, N / n_B)  # wagi SRS

print(f"\nŚrednia w populacji mu_y: {mu_true:.3f}")
print(f"Liczebność S_A: {n_A}")
print(f"Liczebność S_B: {n_B}")


# %% Pseudo-IPW: pełny model PS i z karą L1 (LASSO) --------------------------
# Schemat (Chen, Li, Wu 2020 -- weighted logistic):
#   (1) łączymy S_A (R=1) i S_B (R=0 z wagami d_i^B),
#   (2) dopasowujemy logistic regression z karą L1 i wagami,
#   (3) wyznaczamy pi_hat dla jednostek z S_A i Hajek IPW.

X_pool = np.vstack([X_A, X_B])
R_pool = np.concatenate([np.ones(n_A), np.zeros(n_B)])
w_pool = np.concatenate([np.ones(n_A), d_B])


def hajek(y, w):
    return (w * y).sum() / w.sum()


# Pełny model PS (bez kary)
lr_full = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
lr_full.fit(X_pool, R_pool, sample_weight=w_pool)
pi_hat_full = np.clip(lr_full.predict_proba(X_A)[:, 1], 1e-3, 1 - 1e-3)
mu_ipw_full = hajek(y_A, 1 / pi_hat_full)
print(f"\nIPW pełny (Hajek, 20 zmiennych): {mu_ipw_full:.3f}")

# IPW + LASSO dla różnych wartości C (C = 1/lambda; mniejsze C = silniejsza kara)
print("\nIPW + LASSO dla różnych poziomów regularyzacji:")
results_b = []
for C in [0.0005, 0.001, 0.005, 0.01, 0.05]:
    lr_l1 = LogisticRegression(penalty="l1", C=C, solver="saga",
                                max_iter=10000)
    lr_l1.fit(X_pool, R_pool, sample_weight=w_pool)
    coefs = lr_l1.coef_.ravel()
    sel = np.where(coefs != 0)[0]
    pi_hat = np.clip(lr_l1.predict_proba(X_A)[:, 1], 1e-3, 1 - 1e-3)
    mu = hajek(y_A, 1 / pi_hat)
    results_b.append({"C": C, "n_sel": len(sel),
                      "mu_ipw": round(mu, 3),
                      "sel": sel.tolist()})

print(pd.DataFrame(results_b).to_string(index=False))

print(f"\nNaiwna średnia z S_A: {y_A.mean():.3f}")
print(f"Prawdziwa mu_y: {mu_true:.3f}")
print("\nOczekiwane: LASSO przy odpowiednim C powinien zachować zmienne 2 "
      "i 3 (= x3, x4 -- sterujące selekcją).")


# %% ============================================================
# %% CZĘŚĆ C: Realne dane admin / jvs
# %% ============================================================
admin = pd.read_csv("../data/admin.csv")
jvs   = pd.read_csv("../data/jvs.csv")

# konwersja region do string z zero-padding (jak w R)
admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"]   = jvs["region"].astype(str).str.zfill(2)

# dummy encoding -- po nim mamy >> 4 kolumn, więc dobór zmiennych ma sens
X_admin = pd.get_dummies(admin[["size", "nace", "region", "private"]],
                          drop_first=True).astype(float)
X_jvs   = pd.get_dummies(jvs[["size", "nace", "region", "private"]],
                          drop_first=True).astype(float)
X_admin, X_jvs = X_admin.align(X_jvs, join="outer", axis=1, fill_value=0)

print(f"\nLiczba kolumn macierzy po dummy encoding: {X_admin.shape[1]}")

X_pool_real = np.vstack([X_admin.values, X_jvs.values])
R_pool_real = np.concatenate([np.ones(len(X_admin)), np.zeros(len(X_jvs))])
w_pool_real = np.concatenate([np.ones(len(X_admin)), jvs["weight"].values])

print("\nIPW + LASSO dla single_shift (różne C):")
for C in [0.0005, 0.001, 0.01, 0.1]:
    lr_real = LogisticRegression(penalty="l1", C=C, solver="saga",
                                  max_iter=10000)
    lr_real.fit(X_pool_real, R_pool_real, sample_weight=w_pool_real)
    n_sel = (lr_real.coef_.ravel() != 0).sum()
    pi_hat = np.clip(lr_real.predict_proba(X_admin.values)[:, 1],
                      1e-4, 1 - 1e-4)
    mu = hajek(admin["single_shift"].values, 1 / pi_hat)
    print(f"  C={C:.4f}  zachowanych={n_sel:>2d}/{X_admin.shape[1]}  "
          f"mu={mu:.3f}")

print(f"\nNaiwna średnia z admin: {admin['single_shift'].mean():.3f}")
print("\nSCAD/MCP z wagami (ważona regresja) -- patrz wersja R "
      "(codes/R/09-varsel.R), pakiet nonprobsvy.")
