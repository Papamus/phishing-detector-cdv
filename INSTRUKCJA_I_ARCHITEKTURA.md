# Filtr antyphishingowy: instrukcja i mapa projektu

## Zasada działania

Rozwiązanie klasyfikuje teksty, adresy URL i fragmenty HTML jako benign (`0`) albo phishing (`1`). Całość działa lokalnie na CPU; nie wywołuje API modeli. W notebooku/skrypcie treningowym pobierany jest oryginalny plik `combined_reduced.json` z tego samego adresu Hugging Face co wcześniej. Plik jest cache'owany w `data/`; jego zawartość nie jest przekształcana przed podziałem i treningiem.

Pipeline ogranicza wejście do 4096 znaków, tworzy dwa rzadkie wektory haszujące (n-gramy słowne 1–2 i znakowe 3–4) oraz 20 ręcznych cech strukturalnych. Cechy obejmują m.in. długość i entropię tekstu, sygnały HTML/JavaScript, adres IP, punycode, strukturę hosta, subdomeny i podejrzane słowa w domenie/ścieżce. Liniowy `SGDClassifier(loss="log_loss")` zwraca prawdopodobieństwo phishingu; zapisany próg zamienia je na etykietę.

Podział danych jest stratyfikowany i deterministyczny (`seed=42`): 80% trening, 20% test. Z części treningowej wydzielane jest 10% do kalibracji progu, z celem recall 94%. Test pozostaje odłożony do końcowej ewaluacji. Cel recall na walidacji nie gwarantuje takiego samego wyniku na teście.

## Uruchomienie na Windows

Uruchom PowerShell w katalogu projektu:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r phishing_detector_requirements.txt
```

Wytrenuj i oceń model:

```powershell
.\.venv\Scripts\python.exe train_and_evaluate.py
```

Pierwsze uruchomienie pobierze około 521 MB danych i wymaga internetu. Następne treningi używają lokalnego cache i nie pobierają danych ponownie. Wyniki trafiają do `models/` i `results/`.

Uruchom notebook z tego samego katalogu projektu, wybierz kernel `.venv`, a następnie wykonaj komórki po kolei. Notebook uruchamia ten sam skrypt treningowy, a potem pokazuje zapisane metryki i wykres.

Predykcja na pojedynczym tekście:

```powershell
.\.venv\Scripts\python.exe phishing_detector_predict.py --model models/phishing_detector_option_a.joblib --text "https://example.test/login"
```

Predykcja/ewaluacja na pliku JSON:

```powershell
.\.venv\Scripts\python.exe phishing_detector_predict.py --model models/phishing_detector_option_a.joblib --input phishing_detector_data_sample_test.json --output results/sample_predictions.json
```

Dla jednego HTML/TXT można użyć `--file ścieżka\do\pliku.html`. Zmiana ekstraktora wymaga ponownego treningu; predyktor sprawdza wersję artefaktu i odrzuca niezgodny model.

## Pliki i powiązania

| Plik / katalog | Zawartość | Odwołuje się do |
|---|---|---|
| `phishing_detector_phishing_detector.ipynb` | Interfejs notebookowy: uruchomienie treningu, prezentacja JSON z metrykami i wykresu. Nie zawiera własnego pipeline’u. | `train_and_evaluate.py`, `results/metrics.json`, `results/test_results.png` |
| `train_and_evaluate.py` | Pobranie cache danych, podział train/validation/test, kalibracja progu, trening pipeline’u, zapis modelu, metryk i wykresu. | `phishing_detector_phishing_features.py`, `data/combined_reduced.json`, `models/`, `results/` |
| `phishing_detector_phishing_features.py` | Wspólny ekstraktor cech, przycięcie tekstu, haszujące wektoryzatory i konstruktor pipeline’u. To jedyne źródło definicji modelu. | NumPy i scikit-learn; importowany przez trening i predyktor |
| `phishing_detector_predict.py` | Wczytuje artefakt, sprawdza jego wersję i wykonuje predykcje/ewaluację. | `models/phishing_detector_option_a.joblib`, moduł cech |
| `phishing_detector_requirements.txt` | Pakiety potrzebne do treningu, wykresów i notebooka. | Instalowane do `.venv` przez pip |
| `phishing_detector_data_sample_test.json` | Dołączona mała próbka testowa w formacie rekordów `text`/`label`. | Opcja `--input` predyktora |
| `README_LOCAL.md` | Skrócona instrukcja projektu. | Uzupełnia niniejszy dokument |
| `LICENSE` | Warunki licencyjne repozytorium. | — |
| `data/combined_reduced.json` | Lokalny cache oryginalnego datasetu, około 521 MB. | Wczytywany przez skrypt treningowy |
| `models/phishing_detector_option_a.joblib` | Skompresowany pipeline i próg oraz metadane wersji, źródła, sumy SHA-256 i splitu. | Odczytywany przez predyktor |
| `results/metrics.json` | Metryki ostatniego pełnego testu oraz porównanie z baseline. | Czytany przez notebook |
| `results/metrics_baseline.json` | Zachowane metryki modelu bazowego. | Skrypt używa ich na wykresie porównawczym |
| `results/test_results.png` | Macierz pomyłek i porównanie Recall/Precision/F1. | Wyświetlany w notebooku |
| `results/sample_predictions.json` | Opcjonalny wynik predykcji na próbce. | Tworzony przez `--output` |

## Nazewnictwo i spójność

Moduł cech ma nazwę `phishing_detector_phishing_features.py`, a notebook zachowuje dotychczasową nazwę dla ciągłości projektu. Nazwy flag predyktora (`--model`, `--text`, `--file`, `--input`) opisują dane wejściowe. W skrypcie treningowym tablice `x` i `y` oznaczają teksty i etykiety; ich krótkie nazwy są typowe dla API scikit-learn, natomiast `x_train`, `x_test`, `x_fit`, `x_val` oraz odpowiednie `y_*` oznaczają kolejne rozdziały danych.

Foldery `.venv/`, `data/`, `models/`, `results/`, `.vscode/` i `__pycache__/` są środowiskiem/cache'em/artefaktami lokalnymi, a nie źródłowym kodem. Przy publikacji repozytorium warto ignorować `.venv`, pobrany dataset i `__pycache__`; model i wykres można publikować jako jawne artefakty, jeśli mają być częścią dostawy.

## Ograniczenia pomiarów

`ms_per_sample` to czas predykcji batchowej podzielony przez liczbę próbek; nie jest pomiarem opóźnienia dla pojedynczego wywołania. Wynik na małej paczce 100 przykładów nie jest miarą generalizacji. Zmiana wersji cech wymaga ponownego treningu i testu na pełnym, odłożonym zbiorze.
