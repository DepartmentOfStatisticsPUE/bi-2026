library(ncvreg)     # LASSO, SCAD, MCP w jednym pakiecie
library(nonprobsvy) # wersja >= 0.2.3 -- wspiera vars_selection
library(survey)
set.seed(2026)

n <- 200
p <- 50
X <- matrix(rnorm(n*p), n, p)

# wektor prawdziwych współczynników -- tylko 5 niezerowych
beta_true <- rep(0, p)
beta_true[1]  <-  3.0
beta_true[2]  <-  1.5
beta_true[5]  <-  2.0
beta_true[10] <- -2.5
beta_true[20] <-  1.0

y <- as.numeric(X %*% beta_true) + rnorm(n)
which_active <- which(beta_true != 0)
which_active

fit_lasso <- ncvreg(X, y, penalty = "lasso")
fit_scad  <- ncvreg(X, y, penalty = "SCAD")
fit_mcp   <- ncvreg(X, y, penalty = "MCP")

par(mfrow = c(1, 3))
plot(fit_lasso, main = "LASSO")
plot(fit_scad,  main = "SCAD")
plot(fit_mcp,   main = "MCP")

cv_lasso <- cv.ncvreg(X, y, penalty = "lasso", seed = 2026)
cv_scad  <- cv.ncvreg(X, y, penalty = "SCAD",  seed = 2026)
cv_mcp   <- cv.ncvreg(X, y, penalty = "MCP",   seed = 2026)

cat("Optymalne lambda:\n")
cat("  LASSO:", round(cv_lasso$lambda.min, 4), "\n")
cat("  SCAD: ", round(cv_scad$lambda.min,  4), "\n")
cat("  MCP:  ", round(cv_mcp$lambda.min,   4), "\n")

get_selected <- function(cv_fit) {
  b <- coef(cv_fit)         # wyraz wolny + 50 współczynników
  which(b[-1] != 0)         # indeksy niezerowych (pomijamy intercept)
}

sel_lasso <- get_selected(cv_lasso)
sel_scad  <- get_selected(cv_scad)
sel_mcp   <- get_selected(cv_mcp)

cat("Prawdziwe aktywne zmienne:", which_active, "\n")
cat("LASSO wybrał (", length(sel_lasso), "):", sel_lasso, "\n")
cat("SCAD  wybrał (", length(sel_scad), "): ", sel_scad,  "\n")
cat("MCP   wybrał (", length(sel_mcp), "): ",  sel_mcp,   "\n")

data.frame(
  metoda       = c("LASSO", "SCAD", "MCP"),
  n_wybranych  = c(length(sel_lasso), length(sel_scad), length(sel_mcp)),
  prawdziwe_TP = c(sum(which_active %in% sel_lasso),
                   sum(which_active %in% sel_scad),
                   sum(which_active %in% sel_mcp)),
  falszywe_FP  = c(sum(!sel_lasso %in% which_active),
                   sum(!sel_scad  %in% which_active),
                   sum(!sel_mcp   %in% which_active))
)

N <- 10000
p <- 20
X_pop <- matrix(rnorm(N*p), N, p)

# zmienna y zależy od x1, x2
y_pop <- 1 + 2*X_pop[,1] - 1.5*X_pop[,2] + rnorm(N)
mu_true <- mean(y_pop)

# prawdopodobieństwo włączenia do S_A zależy od x3, x4
eta <- -2 + X_pop[,3] - 0.5*X_pop[,4]
pi_A <- 1/(1 + exp(-eta))
R_A  <- rbinom(N, 1, pi_A)

# próba probabilistyczna S_B: SRS o liczebności n_B = 500
n_B <- 500
idx_B <- sample.int(N, n_B)

cat("Średnia w populacji mu_y:", round(mu_true, 3), "\n")
cat("Liczebność S_A:", sum(R_A), "\n")
cat("Liczebność S_B:", n_B, "\n")

S_A <- data.frame(X_pop[R_A == 1, ], y = y_pop[R_A == 1])
colnames(S_A) <- c(paste0("x", 1:p), "y")

S_B <- data.frame(X_pop[idx_B, ], d = N/n_B) # waga = N/n_B (SRS)
colnames(S_B) <- c(paste0("x", 1:p), "d")

# obiekt survey design dla S_B
sb_svy <- svydesign(ids = ~1, weights = ~d, data = S_B)

mean(S_A$y)

form_pelny <- as.formula(paste("~", paste0("x", 1:p, collapse = " + ")))
ipw_full <- nonprob(data      = S_A,
                    selection = form_pelny,
                    target    = ~ y,
                    svydesign = sb_svy,
                    method_selection = "logit")
ipw_full

ipw_scad <- nonprob(data      = S_A,
                    selection = form_pelny,
                    target    = ~ y,
                    svydesign = sb_svy,
                    method_selection = "logit",
                    control_inference = control_inf(vars_selection = TRUE),
                    control_selection = control_sel(penalty = "SCAD"))
ipw_scad

ipw_mcp <- nonprob(data      = S_A,
                   selection = form_pelny,
                   target    = ~ y,
                   svydesign = sb_svy,
                   method_selection = "logit",
                   control_inference = control_inf(vars_selection = TRUE),
                   control_selection = control_sel(penalty = "MCP"))

ipw_lasso <- nonprob(data      = S_A,
                     selection = form_pelny,
                     target    = ~ y,
                     svydesign = sb_svy,
                     method_selection = "logit",
                     control_inference = control_inf(vars_selection = TRUE),
                     control_selection = control_sel(penalty = "lasso"))

porownanie_b <- rbind(
  "IPW pełny model (20 zm.)" = extract(ipw_full),
  "IPW + SCAD"               = extract(ipw_scad),
  "IPW + MCP"                = extract(ipw_mcp),
  "IPW + LASSO"              = extract(ipw_lasso)
)
porownanie_b
cat("\nPrawdziwa wartość mu_y =", round(mu_true, 3), "\n")

data(admin)
data(jvs)

# obiekt survey design dla S_B
jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)

ipw_real_full <- nonprob(data      = admin,
                         selection = ~ size + nace + region + private,
                         target    = ~ single_shift,
                         svydesign = jvs_svy,
                         method_selection = "logit")

ipw_real_scad <- nonprob(data      = admin,
                         selection = ~ size + nace + region + private,
                         target    = ~ single_shift,
                         svydesign = jvs_svy,
                         method_selection = "logit",
                         control_inference = control_inf(vars_selection = TRUE),
                         control_selection = control_sel(penalty = "SCAD"))

mi_real_scad <- nonprob(data      = admin,
                        outcome   = single_shift ~ size + nace + region + private,
                        svydesign = jvs_svy,
                        method_outcome = "glm",
                        family_outcome = "binomial",
                        control_inference = control_inf(vars_selection = TRUE),
                        control_outcome  = control_out(penalty = "SCAD"))

dr_real_scad <- nonprob(data      = admin,
                        selection = ~ size + nace + region + private,
                        outcome   = single_shift ~ size + nace + region + private,
                        svydesign = jvs_svy,
                        method_outcome = "glm",
                        family_outcome = "binomial",
                        control_inference = control_inf(vars_selection = TRUE),
                        control_selection = control_sel(penalty = "SCAD"),
                        control_outcome  = control_out(penalty = "SCAD"))

rbind(
  "IPW pełny"     = extract(ipw_real_full),
  "IPW + SCAD"    = extract(ipw_real_scad),
  "MI + SCAD"     = extract(mi_real_scad),
  "DR + SCAD"     = extract(dr_real_scad)
)
