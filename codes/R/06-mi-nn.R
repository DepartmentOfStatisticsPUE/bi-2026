library(nonprobsvy) ## wersja >= 0.2.3

data(admin) ## próba nielosowa (S_A)
head(admin)

data(jvs) ## próba losowa (S_B)
head(jvs)

jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)

mi_nn1 <- nonprob(data = admin,
                  outcome = single_shift ~ size,
                  svydesign = jvs_svy,
                  method_outcome = "nn")
mi_nn1

mi_nn2 <- nonprob(data = admin,
                  outcome = single_shift ~ size + nace + region + private,
                  svydesign = jvs_svy,
                  method_outcome = "nn",
                  family_outcome = "binomial")
mi_nn2

mi_pmm1 <- nonprob(data = admin,
                   outcome = single_shift ~ size,
                   svydesign = jvs_svy,
                   method_outcome = "pmm",
                   se = FALSE)
mi_pmm1

mi_pmm2 <- nonprob(data = admin,
                   outcome = single_shift ~ size + nace + region + private,
                   svydesign = jvs_svy,
                   method_outcome = "pmm",
                   se = FALSE,
                   family_outcome = "binomial")
mi_pmm2

rbind(
  "MI-NN  (~size)"          = extract(mi_nn1),
  "MI-NN  (pełny, binomial)" = extract(mi_nn2),
  "MI-PMM (~size)"          = extract(mi_pmm1),
  "MI-PMM (pełny, binomial)" = extract(mi_pmm2)
)
