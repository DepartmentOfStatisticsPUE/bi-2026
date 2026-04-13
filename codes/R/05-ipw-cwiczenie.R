## ----
## Ćwiczenie: IPW krok po kroku (MLE i kalibrowany)
## Autor: Maciej Beręsewicz
##
## UWAGA: w tym ćwiczeniu stosujemy estymator IPW 1 (Horvitza-Thompsona),
## w którym mianownikiem jest N_pop = sum(d_i^B).
## Tak domyślnie działa funkcja nonprob() w R.
## Estymator IPW 2 (Hajeka) z mianownikiem sum(w_i) omówimy osobno.
## ----

## =============================================
## Część A: Dane i estymator naiwny
## =============================================

## Pakiety ----
library(nonprobsvy) ## wersja >= 0.2.3
library(survey)

## Dane ----
data(admin) ## próba nielosowa S_A (CBOP)
data(jvs)   ## próba losowa S_B (badanie popytu na pracę)

head(admin)
head(jvs)

## Obiekt schematu losowania ----
jvs_svy <- svydesign(ids = ~1, weights = ~weight,
                     strata = ~size + nace + region, data = jvs)

## Zadanie A1: Porównaj rozkład zmiennej size w obu źródłach
## Podpowiedź: prop.table(table(...)) dla admin,
## prop.table(svytable(~size, design = jvs_svy)) dla JVS
## Tutaj kod


## Zadanie A2: Oblicz naiwną średnią single_shift z admin (bez korekt).
## Zapisz wynik -- porównasz go z estymatorami IPW w kolejnych częściach.
## Podpowiedź: mean(admin$single_shift)
## Tutaj kod


## =============================================
## Część B: Estymator IPW-MLE
## =============================================

## Przykład: model z jedną zmienną (~size) ----
ipw_mle_simple <- nonprob(
  selection = ~ size,
  target = ~ single_shift,
  svydesign = jvs_svy,
  data = admin,
  method_selection = "logit"
)

extract(ipw_mle_simple)
summary(weights(ipw_mle_simple))
check_balance(~ size - 1, ipw_mle_simple)


## Zadanie B1: Zmodyfikuj powyższy kod tak, aby model wykorzystywał
## WSZYSTKIE zmienne: size + nace + region + private
## Podpowiedź: zmień selection = ~ size na selection = ~ size + nace + region + private
## Tutaj kod


## Zadanie B2: Dla modelu pełnego:
## a) Wypisz rozkład wag: summary(weights(...))
## b) Sprawdź balans: check_balance(~ size - 1, ...)
## Tutaj kod


## =============================================
## Część C: Kalibrowany estymator IPW (GEE)
## =============================================

## Zadanie C1: Napisz model GEE z wszystkimi zmiennymi
## Podpowiedź: skopiuj kod MLE z B1 i dodaj:
## control_selection = control_sel(est_method = "gee")
## Tutaj kod


## Zadanie C2: Porównaj MLE vs GEE
## a) rbind(extract(ipw_mle), extract(ipw_gee))
## b) check_balance(~ size - 1, ...) dla obu
## c) plot(weights(ipw_mle), weights(ipw_gee))
## Tutaj kod


## Zadanie C3: GEE z wartościami globalnymi ----
## Podpowiedź: zamiast svydesign użyj pop_totals, zmień formułę na ~size
pop_totals <- c("(Intercept)" = 51870, sizeM = 13758, sizeS = 29551)
## Tutaj kod


## =============================================
## Część D: Podsumowanie
## =============================================

## Zadanie D1: Uzupełnij tabelę swoimi wynikami
## Zadanie D2: Odpowiedz na pytania:
## a) Jak zmieniły się oszacowania po zastosowaniu IPW vs naiwny?
## b) Kiedy GEE lepszy od MLE?
## c) Kiedy użyjesz pop_totals?
