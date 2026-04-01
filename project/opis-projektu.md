# Badania Internetowe 2025/26 -- zasady zaliczenia

dr Maciej Beręsewicz | Katedra Statystyki, Uniwersytet Ekonomiczny w Poznaniu

---

## 1. Zasady zaliczenia

Zaliczenie składa się z dwóch części:

### Projekt grupowy (50 punktów, max ocena 4.0)

Projekt realizowany w grupach maksymalnie **2--3 osobowych**.

| Punkty     | Ocena |
|------------|-------|
| < 20       | 2.0   |
| [20, 30)   | 3.0   |
| [30, 40)   | 3.5   |
| [40, 50]   | 4.0   |

### Zadania domowe (podniesienie oceny)

Trzy zadania domowe mogą podnieść ocenę o maksymalnie 1 stopień (pod warunkiem, że ocena za projekt wynosi co najmniej 3.0):

| Poprawnie wykonane zadania | Podwyższenie oceny |
|----------------------------|--------------------|
| 1                          | 0.0                |
| 2                          | 0.5                |
| 3                          | 1.0                |

Zadania domowe będą weryfikowane pod kątem **samodzielności wykonania** i zrozumienia rozwiązania.

---

## 2. Projekt zaliczeniowy

### Cel projektu

Projekt jest poświęcony **symulacyjnemu badaniu wybranych estymatorów** prezentowanych w trakcie zajęć, aby odpowiedzieć m.in. na pytania:

- W jakich przypadkach dany estymator(y) znajduje(ą) zastosowanie?
- Jak zachowuje(ą) się dany estymator(y), gdy złamane zostaną założenia?

Projekty wyłącznie odtwarzające wykład są akceptowalne, ale nie na najlepszą ocenę. Preferowana jest własna inwencja twórcza.

### Dane

Projekt bazuje na danych z badania **Bilans Kapitału Ludzkiego (BKL)**, realizowanego przez PARP we współpracy z Uniwersytetem Jagiellońskim.

- **Plik danych**: `Połączona baza danych z badania ludności z lat 2017, 2019 i 2021.sav` (format SPSS, do wczytania w R przez `haven::read_sav()` lub w Pythonie przez `pyreadstat.read_sav()`).
- Zbiór ten traktujemy jako **populację**, dla której znamy rozkład cech $\mathbf{X}$ oraz $Y$.
- **Zmienną celu** ($Y$) studenci wybierają **samodzielnie** na podstawie kwestionariuszy badania (dostępnych w folderze `project/`).
- Zmienne socjo-demograficzne (np. wiek, płeć, wykształcenie, region, status na rynku pracy) pełnią rolę zmiennych pomocniczych $\mathbf{X}$.

### Etapy realizacji

Projekt realizowany jest w trakcie zajęć poprzez oddawanie kolejnych części:

**Etap 1**: Wybór estymatorów i celu badania symulacyjnego

- Określenie zmiennej celu (np. odsetek, średnia, mediana).
- Wstępny dobór estymatorów korygujących błędy nielosowe (spośród omawianych na zajęciach lub z literatury).
- Określenie scenariuszy badania (np. wpływ złamania założeń, duplikaty, błędy pokrycia, słaby model).

**Etap 2**: Implementacja

- Wstępna implementacja badania symulacyjnego (w dowolnym języku -- najlepiej przez Google Colab, które obsługuje R i Pythona).
- Przesłanie notatnika prowadzącemu do weryfikacji.

**Etap 3**: Raport końcowy

- Krótki, **3--4 stronicowy** raport z wynikami (plik HTML, może być Quarto/Jupyter Notebook).

### Przykładowe pytania badawcze

Poniżej kilka propozycji zmiennych celu ($Y$) i pytań, na które można odpowiedzieć w ramach projektu. Zmienne odwołują się do kwestionariusza BKL (numery pytań w nawiasach).

1. **Odsetek osób zatrudnionych na umowę o pracę** (E1) -- $Y \in \{0, 1\}$, estymacja proporcji.
2. **Odsetek osób pracujących na czarno** (N1) -- $Y \in \{0, 1\}$, estymacja proporcji w populacji.
3. **Satysfakcja z pracy** (E17/P8) -- skala 1--5 ("na ile praca odpowiada"), estymacja średniej lub odsetka osób zadowolonych (4--5).
4. **Odsetek osób, które podniosły umiejętności w ciągu 5 lat** (G2.3) -- $Y \in \{0, 1\}$, estymacja proporcji.
5. **Tygodniowy czas pracy** (G1.1) -- zmienna ciągła (godziny/tydzień), estymacja średniej lub mediany.
6. **Odsetek osób pracujących za granicą** (Y1) -- $Y \in \{0, 1\}$, estymacja proporcji.
7. **Zadowolenie z zarobków** (J1.1) -- skala 1--5, estymacja średniej lub odsetka niezadowolonych (1--2).
8. **Odsetek osób pracujących w pracy zgodnej z wykształceniem** (P3.1/E11) -- $Y \in \{0, 1\}$ (odpowiedzi "raczej tak" + "zdecydowanie tak").

Jako zmienne pomocnicze $\mathbf{X}$ można wykorzystać m.in.: płeć (M2), wiek (M1), wykształcenie, region, sektor własności (E6), wielkość zakładu pracy (E8).

### Generowanie prób

Dla zbioru BKL, traktowanego jako populacja, należy utworzyć:

1. **Próbę losową** ($S_B$) -- np. losowanie proste, warstwowe lub proporcjonalne.
2. **Próbę nielosową** ($S_A$) -- przynależność można wygenerować według:
   - modelu regresji logistycznej (logit, probit),
   - modelu nieliniowego (np. uczenie maszynowe),
   - ograniczenia zbioru populacji (np. ucięcie według zmiennej $Y$),
   - innego wybranego podejścia.

### Cel symulacji

Dla wybranego wskaźnika (np. odsetek, średnia, mediana) należy policzyć:

- **Estymator naiwny** -- średnia/odsetek/mediana wyłącznie na podstawie próby nielosowej, bez korekty.
- **Wybrany(e) estymator(y) korygujący(e)** -- spośród omawianych na zajęciach lub własne na podstawie przeglądu literatury.

### Ocena estymatorów

Dla każdego estymatora, na podstawie badania symulacyjnego, należy obliczyć:

- **Obciążenie** (Bias): $\text{Bias}(\hat{\theta}) = \hat{\bar{\theta}} - \theta$, gdzie $\hat{\bar{\theta}} = \sum_{b=1}^{B} \hat{\theta}_b / B$
- **Wariancję** (Variance): $\text{Var}(\hat{\theta}) = \sum_{b=1}^{B} (\hat{\theta}_b - \hat{\bar{\theta}})^2 / B$
- **MSE**: $\text{MSE} = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$

### Pseudo-kod symulacji

1. Dla zbioru BKL określamy wartość prawdziwą cechy $Y$ (np. średnią populacyjną $\theta$).
2. Dla $B$ powtórzeń (np. 100, 500, 1000):
   1. Generujemy przynależność do próby nielosowej $S_A$.
   2. (Opcjonalnie) Generujemy próbę losową $S_B$.
   3. Obliczamy estymator naiwny.
   4. Obliczamy estymator(y) korygujący(e).
   5. Zapisujemy wyniki.
3. Na podstawie $B$ wyników obliczamy Bias, Var, MSE.

### Opcjonalne rozszerzenia

- Wpływ źle wyspecyfikowanego modelu przynależności ($R$) lub modelu $Y$ na oszacowania.
- Wpływ duplikatów w zbiorze danych.
- Wybór podzbioru zmiennych $\mathbf{X}$ (metody doboru zmiennych do modelu vs. uwzględnienie wszystkich).

### Kryteria oceny projektu

| Aspekt | Punkty | Szczegóły |
|--------|--------|-----------|
| **Metodyka** | 15 | Ile estymatorów i jak skomplikowane: jeden wymagający (do 15 pkt), jeden prosty z zajęć (do 10 pkt), 2--3 estymatory z zajęć (do 15 pkt) |
| **Implementacja** | 25 | Poprawność (10 pkt), klarowność kodu (10 pkt), odtwarzalność wyników (5 pkt) |
| **Raport** | 10 | Jasność przekazu (5 pkt), jakość prezentacji wyników i wykresów (5 pkt) |
| **Razem** | **50** | |

### Struktura raportu

Raport powinien zawierać:

1. **Wstęp** -- omówienie celu symulacji
2. **Metodyka** -- omówienie wybranych estymatorów
3. **Opis symulacji** -- cel, założenia, pseudo-kod
4. **Symulacja** -- kod wraz z opisem
5. **Wyniki** -- opis wyników z interpretacją (tabele, wykresy)
6. **Podsumowanie** -- krótkie podsumowanie uzyskanych wyników

---

## 3. Materiały

### Szablon projektu

Raport należy przygotować w oparciu o szablon dostępny w folderze `project/`:

- **Quarto**: [`szablon.qmd`](szablon.qmd) -- szablon z sekcjami: Autorzy, Wprowadzenie, Symulacja (założenia, ocena estymatorów, kod), Wyniki, Podsumowanie.
- **Jupyter Notebook**: [`szablon.ipynb`](szablon.ipynb) -- wersja notebookowa tego samego szablonu.

### Przykładowy projekt

W folderze znajduje się również przykładowy projekt z roku 2023/24 ([`projekt-przyklad.qmd`](projekt-przyklad.qmd) | [HTML](projekt-przyklad.html)), który ilustruje oczekiwany zakres i format raportu. Przykład bazuje na danych o gospodarstw domowych (`gospodarstwa.xlsx`) i porównuje estymator naiwny z estymatorem masowej imputacji (MI) dla zmiennej wydatków.

### Dokumentacja BKL

| Plik | Opis |
|------|------|
| `bkl-raport-metodologiczny.pdf` | Raport metodologiczny BKL 2016--2023 (opis populacji, operatu, doboru próby, wag) |
| `kwestionariusz-2017.pdf` | Kwestionariusz badania ludności BKL 2017 |
| `kwestionariusz-2019.pdf` | Kwestionariusz badania ludności BKL 2019 |
| `kwestionariusz-2021.pdf` | Kwestionariusz badania ludności BKL 2021 |

### Dane

Dane BKL do pobrania: [Połączona baza danych z badania ludności z lat 2017, 2019 i 2021.sav](https://www.parp.gov.pl/images/publications/BKL/Poczona_baza_danych_z_badania_ludnoci_z_lat_20172019_i_2021.sav) (strona projektu: [BKL -- PARP](https://www.parp.gov.pl/component/site/site/bilans-kapitalu-ludzkiego#metodologiabadaniabkl)).
