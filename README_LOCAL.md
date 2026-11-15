# Lokalny filtr antyphishingowy

Model działa lokalnie na CPU i nie korzysta z API modeli. Dane są pobierane z tego samego źródła co w notebooku (`combined_reduced.json`) i cache'owane w `data/`.

## Instalacja i uruchomienie (Windows / PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r phishing_detector_requirements.txt
.\.venv\Scripts\python.exe train_and_evaluate.py
```

Repozytorium zawiera gotowy model w `models/phishing_detector_option_a.joblib`, więc predykcja nie wymaga pobierania danych ani treningu. Trening odtwarza model i zapisuje metryki w `results/metrics.json` oraz wykres w `results/test_results.png`. `data/combined_reduced.json` jest cache'em niezmienionych danych notebooka (około 500 MB); pierwszy trening pobiera go z internetu. Notebook wywołuje ten sam skrypt, więc nie ma osobnej implementacji treningu.

Predykcja na przykładowych danych:

```powershell
.\.venv\Scripts\python.exe phishing_detector_predict.py --model models/phishing_detector_option_a.joblib --input phishing_detector_data_sample_test.json --output results/sample_predictions.json
```

Pojedynczy tekst:

```powershell
.\.venv\Scripts\python.exe phishing_detector_predict.py --model models/phishing_detector_option_a.joblib --text "https://example.test/login"
```

`train_and_evaluate.py` jest jedynym źródłem logiki treningu/evaluacji; ekstraktor jest wspólny z predyktorem. Model zawiera wersję artefaktu i SHA-256 zbioru. Po zmianie cech należy ponownie trenować model; predyktor odrzuci stary, niezgodny artefakt.

Pełna instrukcja uruchomienia, zasada działania i mapa plików znajdują się w [INSTRUKCJA_I_ARCHITEKTURA.md](INSTRUKCJA_I_ARCHITEKTURA.md).
