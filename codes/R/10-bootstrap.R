library(boot)        # klasyczny bootstrap
library(nonprobsvy)  # >= 0.2.3 -- wbudowany bootstrap dla IPW/MI/DR
library(survey)
set.seed(2026)

n <- 100
contam <- rbinom(n, 1, 0.1)               # 10% odstających
x <- rnorm(n, mean = 5, sd = ifelse(contam, 5, 1))
hist(x, breaks = 20, main = "Próba z odstającymi", col = "lightblue")
abline(v = mean(x), col = "red", lwd = 2)

xbar  <- mean(x)
se_cl <- sd(x)/sqrt(n)
cat("Średnia:", round(xbar, 3), "\n")
cat("Klasyczne SE:", round(se_cl, 3), "\n")
cat("Klasyczne 95% CI: [", round(xbar - 1.96*se_cl, 3),
    ",", round(xbar + 1.96*se_cl, 3), "]\n")

mean_stat <- function(data, idx) mean(data[idx])

B <- 1000
boot_res <- boot(data = x, statistic = mean_stat, R = B)
boot_res

se_boot <- sd(boot_res$t)
cat("Bootstrap SE:", round(se_boot, 3), "\n")

ci_perc <- quantile(boot_res$t, c(0.025, 0.975))
cat("Bootstrap 95% percentile CI: [",
    round(ci_perc[1], 3), ",", round(ci_perc[2], 3), "]\n")

# rozkład replikacji
hist(boot_res$t, breaks = 30, main = "Rozkład replikacji bootstrap",
     xlab = expression(bar(x)^"*"), col = "lightgreen")
abline(v = xbar, col = "red", lwd = 2)

n <- 200
x <- runif(n, 0, 10)
# wariancja błędu rośnie z x -- heteroskedastyczność
sigma_i <- 0.5 + 0.3 * x
y <- 1 + 2*x + rnorm(n, sd = sigma_i)

plot(x, y, pch = 16, col = "gray40",
     main = "Regresja z heteroskedastycznym błędem")
abline(coef = c(1, 2), col = "red", lwd = 2)

fit <- lm(y ~ x)
summary(fit)$coefficients

beta_stat <- function(data, idx) {
  m <- lm(y ~ x, data = data[idx, ])
  coef(m)
}

dat <- data.frame(x = x, y = y)
boot_lm <- boot(dat, beta_stat, R = 1000)
boot_lm

se_boot_lm <- apply(boot_lm$t, 2, sd)
data.frame(
  parametr = c("intercept", "x"),
  est        = coef(fit),
  se_klasyczne = sqrt(diag(vcov(fit))),
  se_bootstrap = se_boot_lm
)

N <- 5000
p <- 5
X_pop <- matrix(rnorm(N*p), N, p)
y_pop <- 1 + 2*X_pop[,1] - 1.5*X_pop[,2] + rnorm(N)
mu_true <- mean(y_pop)

# selekcja zależy od x3, x4
eta  <- -2 + X_pop[,3] - 0.5*X_pop[,4]
pi_A <- 1/(1 + exp(-eta))
R_A  <- rbinom(N, 1, pi_A)

# próba probabilistyczna S_B: SRS o n_B = 300
n_B <- 300
idx_B <- sample.int(N, n_B)

S_A <- data.frame(X_pop[R_A == 1, ], y = y_pop[R_A == 1])
colnames(S_A) <- c(paste0("x", 1:p), "y")
S_B <- data.frame(X_pop[idx_B, ], d = N/n_B)
colnames(S_B) <- c(paste0("x", 1:p), "d")

sb_svy <- svydesign(ids = ~1, weights = ~d, data = S_B)

cat("mu_y =", round(mu_true, 3), "; |S_A| =", sum(R_A), "; |S_B| =", n_B, "\n")

ipw_analyt <- nonprob(data      = S_A,
                      selection = ~ x1 + x2 + x3 + x4 + x5,
                      target    = ~ y,
                      svydesign = sb_svy,
                      control_inference = control_inf(var_method = "analytic"))
ipw_analyt

ipw_boot <- nonprob(data      = S_A,
                    selection = ~ x1 + x2 + x3 + x4 + x5,
                    target    = ~ y,
                    svydesign = sb_svy,
                    control_inference = control_inf(var_method = "bootstrap",
                                                    num_boot = 200))
ipw_boot

rbind(
  "IPW analityczne" = extract(ipw_analyt),
  "IPW bootstrap"   = extract(ipw_boot)
)
cat("\nPrawdziwa wartość mu_y =", round(mu_true, 3), "\n")

if (!is.null(ipw_boot$boot_sample)) {
  hist(ipw_boot$boot_sample, breaks = 30,
       main = "Rozkład estymatora IPW w replikacjach bootstrap",
       xlab = expression(hat(mu)[y]^"*"), col = "lightblue")
  abline(v = ipw_boot$output$mean, col = "red", lwd = 2)
}

data(admin)
data(jvs)

jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)

ipw_real_a <- nonprob(data      = admin,
                      selection = ~ size + nace + region + private,
                      target    = ~ single_shift,
                      svydesign = jvs_svy,
                      method_selection = "logit",
                      control_inference = control_inf(var_method = "analytic"))

ipw_real_b <- nonprob(data      = admin,
                      selection = ~ size + nace + region + private,
                      target    = ~ single_shift,
                      svydesign = jvs_svy,
                      method_selection = "logit",
                      control_inference = control_inf(var_method = "bootstrap",
                                                      num_boot = 200))

mi_real_b <- nonprob(data      = admin,
                     outcome   = single_shift ~ size + nace + region + private,
                     svydesign = jvs_svy,
                     method_outcome = "glm",
                     family_outcome = "binomial",
                     control_inference = control_inf(var_method = "bootstrap",
                                                     num_boot = 200))

dr_real_b <- nonprob(data      = admin,
                     selection = ~ size + nace + region + private,
                     outcome   = single_shift ~ size + nace + region + private,
                     svydesign = jvs_svy,
                     method_outcome = "glm",
                     family_outcome = "binomial",
                     control_inference = control_inf(var_method = "bootstrap",
                                                     num_boot = 200))

rbind(
  "IPW analityczne" = extract(ipw_real_a),
  "IPW bootstrap"   = extract(ipw_real_b),
  "MI bootstrap"    = extract(mi_real_b),
  "DR bootstrap"    = extract(dr_real_b)
)
