## ----
## Web-scraping
## Autor: Maciej Beręsewicz
## ----

## Pakiety ----
import requests
from bs4 import BeautifulSoup
import json
import re
import pandas as pd
from lxml import etree

## Archiwum pracuj.pl ----

## Pobieranie źródła strony
response = requests.get("https://archiwum.pracuj.pl/archive/offers?Year=2026&Month=1&PageNumber=1")
soup = BeautifulSoup(response.text, "html.parser")
print(f"Status: {response.status_code}, Długość HTML: {len(response.text)}")

## Pobieramy oferty
oferty = soup.select("div.offers_item")
print(f"Liczba ofert na stronie: {len(oferty)}")

## Pobieramy tytuły
tytuly = [el.get_text(strip=True) for el in soup.select("div.offers_item span.offers_item_link_cnt_part")]
print(tytuly[:10])

## Zadanie (na rozgrzewkę) ----
## 1. Proszę wejść na stronę: https://nabory.kprm.gov.pl
## 2. Proszę nacisnąć "szukaj" aby uzyskać dostęp do wszystkich ofert pracy
## 3. Proszę zbadać strukturę strony
## 4. Proszę napisać program, który pobierze:
##    - link do ogłoszenia
##    - numer ogłoszenia
##    - datę umieszczenia ogłoszenia
##    - miejsce pracy ("Urząd")
## 5. Wynik proszę zapisać do ramki danych

## Bardziej zaawansowane przykłady ----
## 1. Pobieranie danych z zawartych w formacie xml
## 2. Pobieranie danych z zawartych w formacie json
## 3. "Dobieranie się" do ukrytych API (Zbadaj -> Network/Sieć)
## 4. requests.get + BeautifulSoup vs requests_html / selenium
