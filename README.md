# Projekt AI: Wykrywanie anomalii giełdowych

Projekt realizuje wykrywanie anomalii na danych giełdowych w oparciu o klasyfikację binarną.

## Zawartość projektu

- `main.py` - główny skrypt uruchamiający pobieranie danych, eksperymenty ML i analizę statystyczną.
- `requirements.txt` - lista zależności Python.
- `src/` - moduły pomocnicze:
  - `data.py` - pobieranie danych z Yahoo Finance i przygotowanie cech.
  - `models.py` - definicje klasyfikatorów i metod resamplingu.
  - `experiment.py` - walidacja krzyżowa, testy Shapiro-Wilka i t-Studenta.
- `figures/` - gotowe wykresy i wizualizacje wyników.
- `prezentacja.html` - prezentacja projektu.
- `.gitignore` - ignoruje wirtualne środowisko, `.idea`, cache Pythona itd.

## Wymagania

- Python 3.10+ (zalecane)
- `pip install -r requirements.txt`

## Uruchomienie

1. Utwórz i aktywuj wirtualne środowisko:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
3. Uruchom główny skrypt:
   ```bash
   python main.py
   ```

## Uwagi

- Skrypt pobiera dane z Yahoo Finance, więc wymaga połączenia z internetem.
- `main.py` wykonuje pełny eksperyment CV i wypisuje wyniki na konsolę.
- Katalog `figures/` zawiera wyniki analizy i wykresy, które można dołączyć do prezentacji.

## Co warto umieścić na GitHubie

- `main.py`
- `requirements.txt`
- `src/`
- `figures/`
- `prezentacja.html`
- `README.md`
- `.gitignore`

## Co wykluczyć z repozytorium

- `venv/` lub inne środowisko wirtualne
- `.idea/` oraz pliki konfiguracji IDE
- pliki tymczasowe Pythona (`__pycache__/`, `*.pyc`)
