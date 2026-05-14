library(nonprobsvy) ## wersja >= 0.2.3

data(admin) ## próba nielosowa (S_A)
head(admin)

data(jvs) ## próba losowa (S_B)
head(jvs)

jvs_svy <- svydesign(ids = ~ 1,
                     weights = ~ weight,
                     strata = ~ size + nace + region,
                     data = jvs)
jvs_svy

## Przykład 1: DR z modelem liniowym (gaussian) -- ten sam zestaw zmiennych
##              w PS i modelu wynikowym
dr_gauss <- nonprob(data = admin,
                    selection = ~ size + nace,
                    outcome = single_shift ~ size + nace,
                    svydesign = jvs_svy,
                    method_outcome = "glm")
dr_gauss

## Przykład 2: DR z modelem logistycznym (binomial) -- bardziej naturalny
##              dla single_shift (0/1)
dr_binom <- nonprob(data = admin,
                    selection = ~ size + nace,
                    outcome = single_shift ~ size + nace,
                    svydesign = jvs_svy,
                    method_outcome = "glm",
                    family_outcome = "binomial")
dr_binom

## Przykład 3: DR z PS estymowanym metodą GEE (kalibrowany IPW + model wynikowy)
dr_gee <- nonprob(data = admin,
                  selection = ~ size + nace,
                  outcome = single_shift ~ size + nace,
                  svydesign = jvs_svy,
                  method_outcome = "glm",
                  family_outcome = "binomial",
                  control_selection = control_sel(est_method = "gee"))
dr_gee

## Przykład 4: DR z pełnym zestawem zmiennych w obu modelach
dr_full <- nonprob(data = admin,
                   selection = ~ size + nace + region + private,
                   outcome = single_shift ~ size + nace + region + private,
                   svydesign = jvs_svy,
                   method_outcome = "glm",
                   family_outcome = "binomial")
dr_full

## Porównanie -- wszystkie warianty DR
rbind(
  "DR  glm gauss   (size+nace)"  = extract(dr_gauss),
  "DR  glm binom   (size+nace)"  = extract(dr_binom),
  "DR  gee binom   (size+nace)"  = extract(dr_gee),
  "DR  glm binom   (pełny)"      = extract(dr_full)
)

## Dla porównania -- IPW i MI-GLM z pełnym zestawem zmiennych
ipw_full <- nonprob(data = admin,
                    selection = ~ size + nace + region + private,
                    target = ~ single_shift,
                    svydesign = jvs_svy,
                    method_selection = "logit")

mi_full <- nonprob(data = admin,
                   outcome = single_shift ~ size + nace + region + private,
                   svydesign = jvs_svy,
                   method_outcome = "glm",
                   family_outcome = "binomial")

rbind(
  "IPW  (pełny)"          = extract(ipw_full),
  "MI-GLM binom (pełny)"  = extract(mi_full),
  "DR   binom  (pełny)"   = extract(dr_full)
)

## ======================================================================
## Zadanie 1: różne specyfikacje modelu wynikowego
## ======================================================================
## Ustal model selekcji jako ~ size + nace + region + private i dopasuj
## cztery estymatory DR z rosnącą specyfikacją modelu wynikowego
## (wszystkie z family_outcome = "binomial"):
##   - DR-A: outcome = single_shift ~ size
##   - DR-B: outcome = single_shift ~ size + nace
##   - DR-C: outcome = single_shift ~ size + nace + region
##   - DR-D: outcome = single_shift ~ size + nace + region + private
##
## Zestawienie wyniki w jednej tabeli za pomocą rbind(extract(...))
## i odpowiedz:
## a) Jak zmienia się mean wraz z dodawaniem zmiennych do modelu wynikowego?
## b) Czy SE maleje monotonicznie? Jeśli nie -- dlaczego?
## c) Porównaj DR-D z czystym IPW (~ size + nace + region + private) --
##    który ma mniejsze SE?

dr_a <- nonprob(data = admin,
                selection = ~ size + nace + region + private,
                outcome = single_shift ~ size,
                svydesign = jvs_svy,
                method_outcome = "glm",
                family_outcome = "binomial")
dr_b <- update(dr_d, outcome = single_shift ~ size + nace)
dr_c <- update(dr_c, outcome = single_shift ~ size + nace + region)
dr_d <- update(dr_b, outcome = single_shift ~ size + nace + region + private)

rbind(
  "DR-A" = extract(dr_a),
  "DR-B" = extract(dr_b),
  "DR-C" = extract(dr_c),
  "DR-D" = extract(dr_d),
  "IPW" = extract(ipw_full)
)

## ======================================================================
## Zadanie 2: różne specyfikacje modelu selekcji
## ======================================================================
## Ustal model wynikowy jako single_shift ~ size + nace + region + private
## (binomial) i zmieniaj tylko model selekcji:
##   - DR-S1: selection = ~ size
##   - DR-S2: selection = ~ size + nace
##   - DR-S3: selection = ~ size + nace + region + private
##
## a) Jak zmienia się mean? Czy efekt jest mniejszy niż w Zadaniu 1?
## b) Sformułuj wniosek: który z dwóch modeli (PS czy wynikowy) ma większy
##    wpływ na oszacowanie DR w tym przykładzie?

dr_s1 <- nonprob(data = admin,
                selection = ~ size,
                outcome = single_shift ~ size + nace + region + private,
                svydesign = jvs_svy,
                method_outcome = "glm",
                family_outcome = "binomial")

dr_s2 <- update(dr_s1, selection = ~ size + nace)
dr_s3 <- update(dr_s1, selection = ~ size + nace + region + private)

rbind(
  "DR-S1" = extract(dr_s1),
  "DR-S2" = extract(dr_s2),
  "DR-S3" = extract(dr_s3)
)

## ======================================================================
## Zadanie 3: celowe zepsucie jednego z modeli
## ======================================================================
## Sprawdź własność podwójnej odporności empirycznie. Wykorzystaj fakt,
## że single_shift w danych admin jest najsilniej powiązany ze zmiennymi
## size i nace.
##
## a) Zły PS, dobry model wynikowy:
##    selection = ~ private,
##    outcome = single_shift ~ size + nace + region + private (binomial).

z1 <- nonprob(data = admin,
              selection = ~ private,
              outcome = single_shift ~ size + nace + region + private,
              svydesign = jvs_svy,
              method_outcome = "glm",
              family_outcome = "binomial")

## b) Dobry PS, zły model wynikowy:
##    selection = ~ size + nace + region + private,
##    outcome = single_shift ~ private (binomial).

z2 <- nonprob(data = admin,
              selection = ~ size + nace + region + private,
              outcome = single_shift ~ private,
              svydesign = jvs_svy,
              method_outcome = "glm",
              family_outcome = "binomial")


## c) Oba złe:
##    selection = ~ private, outcome = single_shift ~ private (binomial).
##

z3 <- nonprob(data = admin,
              selection = ~ private,
              outcome = single_shift ~ private,
              svydesign = jvs_svy,
              method_outcome = "glm",
              family_outcome = "binomial")


## Zestawienie oszacowania mean w tabeli i porównaj z DR-D z Zadania 1
## (oba modele dobre). Czy w wariantach (a) i (b) oszacowania są zbliżone
## do (d)? Co dzieje się w wariancie (c)?

rbind(
  "DR-Z1" = extract(z1),
  "DR-Z2" = extract(z2),
  "DR-Z3" = extract(z3)
)

## ======================================================================
## Zadanie 4: GEE vs MLE w roli PS
## ======================================================================
## Dla zestawu zmiennych ~ size + nace + region + private dopasuj dwa
## estymatory DR (oba z family_outcome = "binomial"):
##   - DR-MLE: domyślny PS (pseudo-MLE).
##   - DR-GEE: PS estymowany metodą GEE
##             (control_selection = control_sel(est_method = "gee")).
##
## a) Porównaj mean, SE oraz przedziały ufności.
## b) Sprawdź check_balance(~ size - 1, ...) dla obu wariantów --
##    czy GEE daje lepszą zgodność rozkładu size?
## c) Narysuj plot(weights(dr_mle), weights(dr_gee)).
##    Co możesz powiedzieć o rozproszeniu wag?
