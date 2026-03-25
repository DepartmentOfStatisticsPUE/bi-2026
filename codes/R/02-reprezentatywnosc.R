## ----eval = FALSE-------------------------------------------------------------
# install.packages(c("ddi", "nonprobsvy", "cobalt", "WeightIt"))


## -----------------------------------------------------------------------------
library(ddi)
library(nonprobsvy) ## tylko do wczytania danych jvs i admin
library(cobalt)
library(WeightIt)


## -----------------------------------------------------------------------------
data(g2016)


## -----------------------------------------------------------------------------
head(g2016[, c("st", "pct_djt_voters", "cces_pct_djt_vv", "tot_votes", "cces_n_vv")])


## -----------------------------------------------------------------------------
rho <- ddc(
  mu    = 62984824 / 136639786,
  muhat = 12284 / 35829,
  N     = 136639786,
  n     = 35829
)
cat("ddc:", rho, "\nddi:", rho^2, "\n")


## -----------------------------------------------------------------------------
g2016$ddc <- with(g2016, ddc(
  mu    = pct_djt_voters,
  muhat = cces_pct_djt_vv,
  N     = tot_votes,
  n     = cces_n_vv
))
g2016$blad <- g2016$cces_pct_djt_vv - g2016$pct_djt_voters
summary(g2016$ddc)


## -----------------------------------------------------------------------------
top10 <- g2016[order(-abs(g2016$ddc)),
               c("st", "pct_djt_voters", "cces_pct_djt_vv", "blad", "ddc")]
top10[1:10, ]


## ----fig.width = 7, fig.height = 6--------------------------------------------
par(mar = c(4.5, 4.5, 2, 1))
plot(g2016$pct_djt_voters, g2016$cces_pct_djt_vv,
     xlab = "Oficjalny odsetek głosów na Trumpa",
     ylab = "Szacunek CCES",
     pch = 19, col = adjustcolor("steelblue", 0.7),
     xlim = c(0, 1), ylim = c(0, 1),
     main = "Oficjalny wynik vs. CCES (2016)")
abline(0, 1, lty = 2, col = "red", lwd = 2)
text(g2016$pct_djt_voters[abs(g2016$blad) > 0.05],
     g2016$cces_pct_djt_vv[abs(g2016$blad) > 0.05],
     g2016$st[abs(g2016$blad) > 0.05],
     pos = 4, cex = 0.7)
legend("topleft", legend = "linia y = x", lty = 2, col = "red", lwd = 2)


## ----fig.width = 8, fig.height = 5--------------------------------------------
g2016_sorted <- g2016[order(g2016$blad), ]
par(mar = c(4.5, 5, 2, 1))
barplot(g2016_sorted$blad,
        names.arg = g2016_sorted$st,
        las = 2, cex.names = 0.6,
        col = ifelse(g2016_sorted$blad < 0, "steelblue", "coral"),
        ylab = "Błąd (CCES - oficjalny)",
        main = "Błąd estymacji CCES po stanach")
abline(h = 0, lwd = 1)


## ----fig.width = 8, fig.height = 5--------------------------------------------
g2016_sorted2 <- g2016[order(g2016$ddc), ]
par(mar = c(4.5, 5, 2, 1))
barplot(g2016_sorted2$ddc,
        names.arg = g2016_sorted2$st,
        las = 2, cex.names = 0.6,
        col = ifelse(g2016_sorted2$ddc < 0, "steelblue", "coral"),
        ylab = expression(hat(rho)[R*","*G]),
        main = "Data Defect Correlation po stanach")
abline(h = 0, lwd = 1)


## ----fig.width = 7, fig.height = 5--------------------------------------------
g2016$z_n <- with(g2016, (cces_pct_djt_vv - pct_djt_voters) /
                    sqrt(cces_pct_djt_vv * (1 - cces_pct_djt_vv) / cces_n_vv))

par(mar = c(4.5, 4.5, 2, 1))
plot(log10(g2016$tot_votes), g2016$z_n,
     xlab = expression(log[10](N[stan])),
     ylab = expression(Z[n] ~ "(Trump)"),
     pch = 19, col = adjustcolor("steelblue", 0.7),
     main = "Paradoks big data: błąd rośnie z wielkością populacji")
rect(par("usr")[1], -2, par("usr")[2], 2, col = adjustcolor("grey", 0.2), border = NA)
points(log10(g2016$tot_votes), g2016$z_n, pch = 19, col = adjustcolor("steelblue", 0.7))
text(log10(g2016$tot_votes), g2016$z_n, g2016$st, pos = 4, cex = 0.55)
abline(h = 0, lty = 1, col = "grey40")

## regresja liniowa
abline(lm(z_n ~ log10(tot_votes), data = g2016), col = "red", lwd = 2, lty = 2)

## loess
lo <- loess(z_n ~ log10(tot_votes), data = g2016)
ox <- order(log10(g2016$tot_votes))
lines(log10(g2016$tot_votes)[ox], predict(lo)[ox], col = "darkgreen", lwd = 2)

legend("bottomleft", legend = c("OLS", "LOESS", "95% CI (|Z| ≤ 2)"),
       col = c("red", "darkgreen", adjustcolor("grey", 0.4)),
       lwd = c(2, 2, 8), lty = c(2, 1, 1),
       bg = adjustcolor("white", 0.8))


## -----------------------------------------------------------------------------
N <- 38e6

par(mar = c(4, 4, 3, 1))

## rejestr 80%
f_share <- 0.8
bias_80 <- function(x, n_size = N * f_share, D0 = (1-f_share)/f_share) {
  sqrt(n_size) * sqrt(D0 * x^2 / (1 - D0 * x^2))
}

curve(bias_80, from = 0.0001, to = 0.006, ylim = c(0, 15),
      ylab = "Błąd standardowy (Z_n)", xlab = expression(rho[R*","*G]),
      main = "Paradoks Big Data (Meng 2018, N = 38 mln)", lwd = 2)

## próba losowa 1%
f_share <- 0.01
bias_01 <- function(x, n_size = N * f_share, D0 = (1-f_share)/f_share) {
  sqrt(n_size) * sqrt(D0 * x^2 / (1 - D0 * x^2))
}

curve(bias_01, from = 0.0001, to = 0.006, add = TRUE, col = "red", lwd = 2)
abline(v = 0.001, h = 6.13, lty = 3, col = "red")
abline(v = 0.005, h = 13.78, lty = 3, col = "black")
legend("bottomright", legend = c("próba losowa (1%)", "rejestr (80%)"),
       lty = 1, col = c("red", "black"), lwd = 2, bg = adjustcolor("white", 0.8))


## -----------------------------------------------------------------------------
data(jvs)
data(admin)


## -----------------------------------------------------------------------------
head(jvs)


## -----------------------------------------------------------------------------
head(admin)


## -----------------------------------------------------------------------------
## ważona tablica częstości (JVS z wagami losowania)
wtab <- function(x, w) {
  tapply(w, x, sum) / sum(w)
}

## rozkład wg wielkości firmy
tab_jvs <- wtab(jvs$size, jvs$weight)
tab_admin <- prop.table(table(admin$size))
rbind("JVS (ważone)" = round(tab_jvs, 3), CBOP = round(tab_admin, 3))


## -----------------------------------------------------------------------------
## rozkład wg sektora (prywatny vs publiczny)
tab_jvs_p <- wtab(jvs$private, jvs$weight)
tab_admin_p <- prop.table(table(admin$private))
rbind("JVS (ważone)" = round(tab_jvs_p, 3), CBOP = round(tab_admin_p, 3))


## ----fig.width = 8, fig.height = 4--------------------------------------------
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

## wykres: wielkość firmy
barplot(rbind(tab_jvs, tab_admin), beside = TRUE,
        col = c("steelblue", "coral"), legend.text = c("JVS (ważone)", "CBOP"),
        main = "Rozkład wg wielkości firmy", ylab = "Odsetek",
        args.legend = list(x = "topright", bg = adjustcolor("white", 0.8)))

## wykres: sektor
barplot(rbind(tab_jvs_p, tab_admin_p), beside = TRUE,
        col = c("steelblue", "coral"),
        names.arg = c("Publiczna", "Prywatna"),
        main = "Rozkład wg sektora", ylab = "Odsetek")


## -----------------------------------------------------------------------------
## wspólne zmienne pomocnicze
wspol <- intersect(names(jvs), names(admin))
wspol <- setdiff(wspol, c("id", "weight", "single_shift"))

jvs_sub <- jvs[, c(wspol, "weight")]
jvs_sub$source <- 0   ## JVS = grupa referencyjna

admin_sub <- admin[, wspol]
admin_sub$weight <- 1  ## CBOP: waga = 1
admin_sub$source <- 1  ## CBOP = próba nielosowa

combined <- rbind(jvs_sub, admin_sub)
combined$source <- factor(combined$source, levels = c(0, 1),
                          labels = c("JVS", "CBOP"))

cat("JVS:", sum(combined$source == "JVS"), "firm\n")
cat("CBOP:", sum(combined$source == "CBOP"), "firm\n")


## -----------------------------------------------------------------------------
bal.tab(source ~ size + private + nace + region,
        data = combined,
        s.weights = combined$weight,
        stats = c("m", "ks"),
        binary = "std")


## -----------------------------------------------------------------------------
W <- weightit(
  source ~ size + private + nace + region,
  data = combined,
  method = "ps",       ## regresja logistyczna
  estimand = "ATT",    ## cel: zrównoważyć JVS do struktury CBOP
  s.weights = combined$weight  ## wagi losowania JVS
)
summary(W)


## -----------------------------------------------------------------------------
bal.tab(W, stats = c("m", "ks"), un = TRUE)


## ----fig.width = 8, fig.height = 6--------------------------------------------
love.plot(W, stats = "mean.diffs", threshold = 0.1,
          abs = TRUE, var.order = "unadjusted",
          title = "Balans zmiennych: CBOP vs JVS",
          colors = c("coral", "steelblue"))


## -----------------------------------------------------------------------------
ps_model <- glm(source == "CBOP" ~ size + private + nace + region,
                data = combined, family = binomial, weights = weight)
ps_hat <- predict(ps_model, type = "response")

R_indicator <- 1 - 2 * sd(ps_hat)
cat("R-indicator:", round(R_indicator, 4), "\n")
cat("SD propensity scores:", round(sd(ps_hat), 4), "\n")


## ----fig.width = 7, fig.height = 4--------------------------------------------
par(mar = c(4.5, 4.5, 2, 1))
hist(ps_hat, breaks = 30, col = "steelblue", border = "white",
     main = "Rozkład propensity scores P(CBOP | X)",
     xlab = "Prawdopodobieństwo przynależności do CBOP",
     ylab = "Liczba obserwacji")
abline(v = mean(ps_hat), lty = 2, col = "red", lwd = 2)
legend("topright", legend = paste("Średnia:", round(mean(ps_hat), 3)),
       lty = 2, col = "red", lwd = 2, bg = adjustcolor("white", 0.8))

