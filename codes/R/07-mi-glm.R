library(nonprobsvy) ## wersja >= 0.2.3

data(admin) ## próba nielosowa (S_A)
head(admin)

data(jvs) ## próba losowa (S_B)
head(jvs)

jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)

## mi glm z jedną zmienną (gaussian)
mi_glm1 <- nonprob(data = admin,
                   outcome = single_shift ~ size,
                   svydesign = jvs_svy,
                   method_outcome = "glm")
mi_glm1

## mi glm z jedną zmienną (binomial)
mi_glm2 <- nonprob(data = admin,
                   outcome = single_shift ~ size,
                   svydesign = jvs_svy,
                   method_outcome = "glm",
                   family_outcome = "binomial")
mi_glm2

## mi glm wszystkie zmienne (binomial) 

mi_glm3 <- nonprob(data = admin,
                   outcome = single_shift ~ size + nace + region + private,
                   svydesign = jvs_svy,
                   method_outcome = "glm",
                   family_outcome = "binomial")
mi_glm3

rbind(
  "MI-GLM gauss  (~size)"            = extract(mi_glm1),
  "MI-GLM binom  (~size)"            = extract(mi_glm2),
  "MI-GLM binom  (pełny)"            = extract(mi_glm3)
)

