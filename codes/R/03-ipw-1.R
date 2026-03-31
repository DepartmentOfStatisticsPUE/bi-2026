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


## Przykład 1: IPW z jedną zmienną -----------------------------------------
## P(R_A = 1 | size)
ipw_est1 <- nonprob(selection = ~ size,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit")

ipw_est1


## -----------------------------------------------------------------------------
plot(ipw_est1)


## -----------------------------------------------------------------------------
summary(ipw_est1)


## -----------------------------------------------------------------------------
extract(ipw_est1)


## -----------------------------------------------------------------------------
weights(ipw_est1) |> table()
1/3.36782061369001
1/4.47997394983112
1/7.92039459072027


## -----------------------------------------------------------------------------
check_balance(~size-1, ipw_est1)
check_balance(~private, ipw_est1)


## -----------------------------------------------------------------------------
coef(ipw_est1)


## Przykład 2: IPW ze wszystkimi zmiennymi ---------------------------------
## P(R_A = 1 | size, nace, region, private)
ipw_est2 <- nonprob(selection = ~ size + nace + region + private,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    data = admin,
                    method_selection = "logit")


rbind(ipw_est1$output, ipw_est2$output)
plot(ipw_est1)


## -----------------------------------------------------------------------------
coef(ipw_est2)


## -----------------------------------------------------------------------------
summary(ipw_est2)


## -----------------------------------------------------------------------------
weights(ipw_est2) |> hist(breaks = "fd")


## -----------------------------------------------------------------------------
check_balance(~size-1,ipw_est1)
check_balance(~size-1,ipw_est2)


## Porównanie rozkładów: model 1 -------------------------------------------
ipw_est1_comp <- as.weightit(x = 1/ipw_est1$ps_scores,
                             treat = ipw_est1$R,
                             covs = rbind(admin[, c("nace", "region", "size", "private")],
                                          jvs[, c("nace", "region", "size", "private")]))
bal.tab(ipw_est1_comp, un = T) |> plot()


## Porównanie rozkładów: model 2 -------------------------------------------
ipw_est2_comp <- as.weightit(x = 1/ipw_est2$ps_scores,
                             treat = ipw_est2$R,
                             covs = rbind(admin[, c("nace", "region", "size", "private")],
                                          jvs[, c("nace", "region", "size", "private")]))
bal.tab(ipw_est2_comp, un = T) |> plot()
