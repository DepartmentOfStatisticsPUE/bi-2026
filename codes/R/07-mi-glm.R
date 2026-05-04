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

## ======================================================================
## Ćwiczenie 1: Porównanie MI-GLM z MI-NN i MI-PMM
## ======================================================================
## Dopasuj cztery modele z pełnym zestawem zmiennych
## (~ size + nace + region + private) i zestawienie wyniki
## w jednej tabeli za pomocą rbind(extract(...)):
##   - MI-NN (method "nn")
##   - MI-PMM z rodziną "binomial" (method "pmm", se = FALSE)
##   - MI-GLM z rodziną "gaussian" (method "glm")
##   - MI-GLM z rodziną "binomial" (method "glm")
##
## Pytania:
## a) Który estymator daje najwęższą wartość SE?
## b) Jak bardzo różnią się oszacowania mean pomiędzy metodami?

## ======================================================================
## Ćwiczenie 2: Rola specyfikacji modelu
## ======================================================================
## Dopasuj cztery modele MI-GLM (family_outcome = "binomial")
## z rosnącą liczbą zmiennych objaśniających:
##   - Model A: ~ private
##   - Model B: ~ private + size
##   - Model C: ~ private + size + nace
##   - Model D: ~ size + nace + region + private (pełny)
##
## Zestawienie wyniki w jednej tabeli i odpowiedz na pytania:
## a) Jak zmienia się oszacowanie mean wraz z dodawaniem zmiennych?
## b) Czy SE maleje monotonicznie? Jeśli nie — dlaczego?

## ======================================================================
## Ćwiczenie 3: Porównanie MI-GLM z estymatorami IPW
## ======================================================================
## Dopasuj estymator IPW z pełnym modelem selekcji
## (selection = ~ size + nace + region + private, method_selection = "logit")
## — por. notatnik 03-ipw-1.R — i zestawienie go z MI-GLM
## (binomial, pełny model) w jednej tabeli.
##
## Pytania:
## a) Które podejście (IPW czy MI-GLM) daje mniejszy SE?
## b) Czy oszacowania mean są zbliżone? Co to mówi o spójności obu metod?
## c) IPW modeluje mechanizm selekcji P(R=1|x), a MI-GLM modeluje
##    zmienną celu E(Y|x). W jakiej sytuacji jedno podejście będzie
##    lepsze od drugiego?

