"""
Estymacja wariancji przez bootstrap
====================================

Wersja Python notatnika 10-bootstrap. Polski.

Sekcje:
- Część A: bootstrap dla średniej (z odstającymi)
- Część B: bootstrap dla regresji (heteroskedastyczność)
- Część C: bootstrap dla pseudo-IPW (sztuczny przykład)
- Część D: szkic dla realnych danych admin/jvs

Uwaga: nonprobsvy nie ma odpowiednika w Pythonie. Dla pełnej analizy
prób nielosowych z wbudowanym bootstrapem -- patrz wersja R.
"""

# %% Pakiety -----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(2026)


# %% ============================================================
# %% CZĘŚĆ A: Bootstrap dla średniej
# %% ============================================================

# Mieszanka: 90% N(5, 1) + 10% N(5, 5)
n = 100
contam = rng.binomial(1, 0.1, size=n)
x = rng.normal(5, np.where(contam == 1, 5.0, 1.0), size=n)

xbar = x.mean()
se_cl = x.std(ddof=1) / np.sqrt(n)
print(f"Średnia: {xbar:.3f}")
print(f"Klasyczne SE: {se_cl:.3f}")
print(f"Klasyczne 95% CI: [{xbar - 1.96*se_cl:.3f}, {xbar + 1.96*se_cl:.3f}]")


# %% Bootstrap dla średniej --------------------------------------------------
B = 1000
boot_means = np.empty(B)
for b in range(B):
    idx = rng.choice(n, size=n, replace=True)
    boot_means[b] = x[idx].mean()

se_boot = boot_means.std(ddof=1)
ci_perc = np.quantile(boot_means, [0.025, 0.975])
print(f"\nBootstrap SE: {se_boot:.3f}")
print(f"Bootstrap 95% percentile CI: [{ci_perc[0]:.3f}, {ci_perc[1]:.3f}]")

# wykres rozkładu replikacji
plt.figure(figsize=(7, 4))
plt.hist(boot_means, bins=30, color="lightgreen", edgecolor="black")
plt.axvline(xbar, color="red", lw=2, label="oryginalna średnia")
plt.title("Rozkład replikacji bootstrap (Część A)")
plt.xlabel(r"$\bar{x}^*$")
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/10-bootstrap-a.png", dpi=80)
plt.close()


# %% ============================================================
# %% CZĘŚĆ B: Bootstrap dla regresji (heteroskedastyczność)
# %% ============================================================

n = 200
x = rng.uniform(0, 10, size=n)
sigma_i = 0.5 + 0.3 * x
y = 1 + 2 * x + rng.normal(0, sigma_i, size=n)

# klasyczne SE z lm
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()
print("\nKlasyczne SE (lm):")
print(ols.summary().tables[1])


# %% Paired bootstrap dla współczynników regresji ----------------------------
B = 1000
boot_coefs = np.empty((B, 2))
for b in range(B):
    idx = rng.choice(n, size=n, replace=True)
    Xb, yb = X[idx], y[idx]
    boot_coefs[b] = sm.OLS(yb, Xb).fit().params

se_boot_lm = boot_coefs.std(ddof=1, axis=0)
df_b = pd.DataFrame({
    "parametr": ["intercept", "x"],
    "est":          ols.params,
    "se_klasyczne": ols.bse,
    "se_bootstrap": se_boot_lm,
})
print("\nPorównanie SE -- regresja z heteroskedastycznością:")
print(df_b.to_string(index=False))


# %% ============================================================
# %% CZĘŚĆ C: Bootstrap dla pseudo-IPW (sztuczny przykład)
# %% ============================================================

# Populacja
N, p = 5000, 5
X_pop = rng.standard_normal((N, p))
y_pop = 1 + 2 * X_pop[:, 0] - 1.5 * X_pop[:, 1] + rng.standard_normal(N)
mu_true = y_pop.mean()

# selekcja zależy od x3, x4
eta = -2 + X_pop[:, 2] - 0.5 * X_pop[:, 3]
pi_A = 1 / (1 + np.exp(-eta))
R_A = rng.binomial(1, pi_A)

n_B = 300
idx_B = rng.choice(N, size=n_B, replace=False)

X_A = X_pop[R_A == 1]
y_A = y_pop[R_A == 1]
n_A = X_A.shape[0]
X_B = X_pop[idx_B]
d_B = np.full(n_B, N / n_B)

print(f"\nmu_y = {mu_true:.3f}  |S_A| = {n_A}  |S_B| = {n_B}")


def fit_ipw(X_A, y_A, X_B, d_B):
    """Pseudo-MLE PS (logit z wagami) -> Hajek IPW."""
    X_pool = np.vstack([X_A, X_B])
    R_pool = np.concatenate([np.ones(len(X_A)), np.zeros(len(X_B))])
    w_pool = np.concatenate([np.ones(len(X_A)), d_B])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(X_pool, R_pool, sample_weight=w_pool)
    pi_hat = np.clip(lr.predict_proba(X_A)[:, 1], 1e-3, 1 - 1e-3)
    w = 1 / pi_hat
    return (w * y_A).sum() / w.sum()


mu_ipw = fit_ipw(X_A, y_A, X_B, d_B)
print(f"IPW (jednorazowe): {mu_ipw:.3f}")


# %% Bootstrap dla IPW -------------------------------------------------------
# Krok 1: ze zwracaniem n_A jednostek z S_A
# Krok 2: ze zwracaniem n_B jednostek z S_B (SRS bootstrap)
# Krok 3: re-fit IPW
B = 200
boot_mu = np.empty(B)
for b in range(B):
    idx_A_b = rng.choice(n_A, size=n_A, replace=True)
    idx_B_b = rng.choice(n_B, size=n_B, replace=True)
    boot_mu[b] = fit_ipw(X_A[idx_A_b], y_A[idx_A_b],
                          X_B[idx_B_b], d_B[idx_B_b])

se_boot_ipw = boot_mu.std(ddof=1)
ci_boot_ipw = np.quantile(boot_mu, [0.025, 0.975])
print(f"Bootstrap SE (IPW): {se_boot_ipw:.4f}")
print(f"Bootstrap 95% CI:   [{ci_boot_ipw[0]:.3f}, {ci_boot_ipw[1]:.3f}]")
print(f"Prawdziwa mu_y:     {mu_true:.3f}")


# %% ============================================================
# %% CZĘŚĆ D: Realne dane admin / jvs -- szkic
# %% ============================================================
admin = pd.read_csv("../data/admin.csv")
jvs   = pd.read_csv("../data/jvs.csv")

admin["region"] = admin["region"].astype(str).str.zfill(2)
jvs["region"]   = jvs["region"].astype(str).str.zfill(2)

X_admin = pd.get_dummies(admin[["size", "nace", "region", "private"]],
                          drop_first=True).astype(float).values
X_jvs   = pd.get_dummies(jvs[["size", "nace", "region", "private"]],
                          drop_first=True).astype(float)
X_admin_df = pd.get_dummies(admin[["size", "nace", "region", "private"]],
                             drop_first=True).astype(float)
X_admin_df, X_jvs = X_admin_df.align(X_jvs, join="outer", axis=1, fill_value=0)
X_admin = X_admin_df.values
X_jvs   = X_jvs.values
y_admin = admin["single_shift"].values
d_jvs   = jvs["weight"].values
n_adm   = len(admin)
n_jvs   = len(jvs)


def fit_ipw_real(X_A, y_A, X_B, d_B):
    X_pool = np.vstack([X_A, X_B])
    R_pool = np.concatenate([np.ones(len(X_A)), np.zeros(len(X_B))])
    w_pool = np.concatenate([np.ones(len(X_A)), d_B])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(X_pool, R_pool, sample_weight=w_pool)
    pi_hat = np.clip(lr.predict_proba(X_A)[:, 1], 1e-4, 1 - 1e-4)
    w = 1 / pi_hat
    return (w * y_A).sum() / w.sum()


mu_real = fit_ipw_real(X_admin, y_admin, X_jvs, d_jvs)
print(f"\nIPW na admin/jvs (single_shift): {mu_real:.3f}")

# Bootstrap (mniej replikacji ze względu na koszt obliczeniowy)
B = 100
boot_real = np.empty(B)
for b in range(B):
    idx_A = rng.choice(n_adm, size=n_adm, replace=True)
    idx_B = rng.choice(n_jvs, size=n_jvs, replace=True)
    boot_real[b] = fit_ipw_real(X_admin[idx_A], y_admin[idx_A],
                                  X_jvs[idx_B], d_jvs[idx_B])

se_real = boot_real.std(ddof=1)
ci_real = np.quantile(boot_real, [0.025, 0.975])
print(f"Bootstrap SE (B={B}):  {se_real:.4f}")
print(f"Bootstrap 95% CI:    [{ci_real[0]:.3f}, {ci_real[1]:.3f}]")
print(f"Naiwna średnia z admin: {y_admin.mean():.3f}")
print("\nDla MI/DR oraz bardziej precyzyjnego schematu bootstrap (Rao-Wu) "
      "patrz wersja R (codes/R/10-bootstrap.R, pakiet nonprobsvy).")
