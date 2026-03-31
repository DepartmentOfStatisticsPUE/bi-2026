## ----eval = FALSE-------------------------------------------------------------
# install.packages(c("nonprobsvy", "cobalt", "WeightIt"))


## -----------------------------------------------------------------------------
library(nonprobsvy) ## wersja 0.2.3
library(cobalt)
library(WeightIt)


## -----------------------------------------------------------------------------
data(admin) ## próba nielosowa (S_A)
head(admin)


## -----------------------------------------------------------------------------
data(jvs) ## próba losowa (S_B)
head(jvs)


## -----------------------------------------------------------------------------
jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)


## Przykład 1: MLE vs GEE z jedną zmienną ----------------------------------
## P(R_A = 1 | size)

## standardowy IPW (MLE)
ipw_mle <- nonprob(selection = ~ size,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit")

ipw_mle


## GEE
ipw_gee <- nonprob(selection = ~ size,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit",
                    control_selection = control_sel(est_method = "gee"))

ipw_gee


## -----------------------------------------------------------------------------
check_balance(~size-1, ipw_mle)
check_balance(~size-1, ipw_gee)


## -----------------------------------------------------------------------------
coef(ipw_mle)


## -----------------------------------------------------------------------------
coef(ipw_gee)


## Przykład 2: MLE vs GEE ze wszystkimi zmiennymi --------------------------
## P(R_A = 1 | size, nace, region, private)

## standardowy IPW (MLE)
ipw_mle2 <- nonprob(selection = ~ size + nace + region + private,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit")


## GEE
ipw_gee2 <- nonprob(selection = ~ size + nace + region + private,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit",
                    control_selection = control_sel(est_method = "gee"))


## Porównanie oszacowań
rbind(
  extract(ipw_mle2),
  extract(ipw_gee2)
)


## Porównanie wag
plot(x = weights(ipw_mle2),
     y = weights(ipw_gee2),
     xlab = "Wagi IPW", ylab = "Wagi GEE")


## -----------------------------------------------------------------------------
summary(weights(ipw_mle2))
summary(weights(ipw_gee2))


## Metoda GEE zapewnia zgodność rozkładów
check_balance(~size-1,ipw_mle2)
check_balance(~size-1,ipw_gee2)


## Przykład 3: znamy tylko wartości globalne --------------------------------
pop_totals <- c("(Intercept)"=51870, sizeM = 13758, sizeS = 29551)


## -----------------------------------------------------------------------------
ipw_gee3 <- nonprob(selection = ~ size,
                    target = ~ single_shift,
                    pop_totals = pop_totals,
                    data = admin,
                    method_selection = "logit",
                    control_selection = control_sel(est_method = "gee"))

ipw_gee3


## Porównanie GEE z danymi jednostkowymi vs wartościami globalnymi
rbind(
  extract(ipw_gee),
  extract(ipw_gee3)
)
