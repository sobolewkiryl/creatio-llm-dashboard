# Creatio · LLM Brand Visibility Dashboard

Dashboard do śledzenia widoczności brandu Creatio w odpowiedziach LLM (ChatGPT, Gemini, Copilot), oparty na eksporcie z Ahrefs Brand Radar.

---

## Struktura plików

```
creatio-llm-dashboard/
├── app.py              ← główna aplikacja Streamlit
├── clusters.csv        ← mapowanie promptów na klastry (edytujesz tutaj)
├── requirements.txt    ← biblioteki Python
├── data/               ← tutaj zapisują się snapshoty (CSV per okres)
│   └── .gitkeep
└── README.md
```

---

## Jak używać

### Co dwa tygodnie (workflow):

1. Pobierz eksport CSV z **Ahrefs Brand Radar** (pełny eksport z odpowiedziami)
2. Otwórz dashboard pod linkiem Streamlit Cloud
3. Wgraj plik w panelu bocznym → kliknij **Generuj raport**
4. Dashboard pokazuje coverage per klaster, per LLM, per kraj
5. Porównanie z poprzednim okresem działa automatycznie
6. Snapshot zapisuje się w folderze `data/` — **wgraj go na GitHub** żeby był dostępny dla wszystkich

### Jak wgrać snapshot na GitHub:
Po kliknięciu "Generuj raport", pobierz snapshot przyciskiem na dole strony, a następnie wgraj plik do folderu `data/` w repozytorium przez interfejs GitHub (Add file → Upload files).

---

## Aktualizacja klastrów

Jeśli Ahrefs doda nowe prompty, których nie ma w `clusters.csv`:
- Dashboard pokaże ostrzeżenie z listą niesklasyfikowanych promptów
- Dodaj je do `clusters.csv` (kolumna `Prompt` lowercase, kolumna `Tags` = nazwa klastra)
- Wgraj zaktualizowany plik na GitHub — dashboard od razu go podchwyci

---

## Setup (jednorazowo)

### 1. GitHub
- Utwórz nowe repozytorium (może być prywatne)
- Wgraj wszystkie pliki z tego folderu

### 2. Streamlit Cloud
- Wejdź na [share.streamlit.io](https://share.streamlit.io)
- Zaloguj się przez GitHub
- Kliknij **New app**
- Wybierz repozytorium, branch `main`, plik `app.py`
- Kliknij **Deploy** — za ~2 minuty masz działający link

---

## Wymagania

Python 3.10+, biblioteki w `requirements.txt` (instalowane automatycznie przez Streamlit Cloud).
