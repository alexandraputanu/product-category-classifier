# Product Category Classifier

Acest proiect rezolvă sarcina de clasificare automată a produselor pe categorii folosind titlul produsului.

Modelul final primește un text precum:

```text
iphone 7 32gb gold
```

și prezice categoria:

```text
Mobile Phones
```


## GitHub repository recomandat

Repository public recomandat:

```text
https://github.com/alexandraputanu/product-category-classifier
```

Nume recomandat pentru repository:

```text
product-category-classifier
```

Descriere recomandată:

```text
Machine learning project for automatic product category classification based on product titles.
```

## Upload direct din GitHub, fără terminal

Dacă lucrezi doar din GitHub:

1. Intră pe GitHub.
2. Apasă **New repository**.
3. Repository name: `product-category-classifier`.
4. Alege **Public**.
5. Nu bifa opțiunea de README, pentru că proiectul are deja `README.md`.
6. Apasă **Create repository**.
7. Apasă **uploading an existing file**.
8. Încarcă folderele și fișierele din proiect:
   - `data/`
   - `models/`
   - `notebooks/`
   - `reports/`
   - `src/`
   - `.gitignore`
   - `README.md`
   - `requirements.txt`
9. Apasă **Commit changes**.

La final, linkul de predare va fi:

```text
https://github.com/alexandraputanu/product-category-classifier
```

## Structura proiectului

```bash
product-category-classifier/
│
├── data/
│   └── products.csv
│
├── models/
│   └── product_classifier.pkl
│
├── notebooks/
│   └── product_category_classification.ipynb
│
├── reports/
│   └── model_report.txt
│
├── src/
│   ├── train_model.py
│   └── predict_category.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

Fișierul folosit este `data/products.csv`.

Coloane importante:

- `Product Title` – textul folosit pentru predicție
- `Category Label` – categoria reală, folosită ca target

În dataset există spații în unele nume de coloane, de aceea scriptul aplică:

```python
df.columns = df.columns.str.strip()
```

## Instalare

Clonează repository-ul și instalează dependențele:

```bash
pip install -r requirements.txt
```

## Antrenarea modelului

Rulează:

```bash
python src/train_model.py
```

Scriptul:

1. încarcă dataset-ul;
2. curăță valorile lipsă;
3. creează exemple suplimentare prin feature engineering text;
4. compară două modele:
   - Linear SVM
   - Multinomial Naive Bayes
5. salvează cel mai bun model în `models/product_classifier.pkl`;
6. salvează raportul în `reports/model_report.txt`.

## Testare interactivă

Rulează:

```bash
python src/predict_category.py
```

Apoi introdu un titlu de produs:

```text
Enter product title: iphone 7 32gb gold
Predicted category: Mobile Phones
```

Pentru oprire:

```text
quit
```

## Rezultate obținute

Pe split-ul de testare, modelul final selectat a obținut aproximativ:

```text
Linear SVM accuracy: 0.9811
Multinomial Naive Bayes accuracy: 0.9387
```

Modelul final ales este **Linear SVM**, deoarece a avut cea mai bună acuratețe.

## Exemple manuale

```text
iphone 7 32gb gold -> Mobile Phones
olympus e m10 mark iii geh use silber -> Digital Cameras
kenwood k20mss15 solo -> Microwaves
bosch wap28390gb 8kg 1400 spin -> Washing Machines
bosch serie 4 kgv39vl31g -> Fridge Freezers
smeg sbs8004po -> Fridge Freezers
```

## Decizii tehnice

### De ce TF-IDF?

TF-IDF este potrivit pentru clasificarea textelor scurte deoarece transformă cuvintele și fragmentele de caractere în valori numerice utile pentru modelele ML.

### De ce am folosit și n-gramuri pe caractere?

Titlurile produselor conțin coduri precum `sbs8004po`, `kgv39vl31g` sau `wap28390gb`. N-gramurile pe caractere ajută modelul să învețe tipare din aceste coduri, nu doar din cuvinte complete.

### De ce Linear SVM?

Linear SVM funcționează foarte bine pentru clasificarea textului cu multe features sparse generate de TF-IDF.

## Feature engineering

Pe lângă textul original, scriptul creează exemple suplimentare prin eliminarea cuvintelor din categoria produsului din titlu.

Exemplu:

```text
smeg sbs8004po fridge freezer -> smeg sbs8004po
```

Această augmentare ajută modelul să prezică mai bine categoria când utilizatorul introduce doar codul sau o variantă scurtă a titlului.

## Fișiere importante

- `src/train_model.py` – antrenează, evaluează și salvează modelul
- `src/predict_category.py` – script interactiv de predicție
- `models/product_classifier.pkl` – modelul antrenat
- `notebooks/product_category_classification.ipynb` – analiză completă și dezvoltare
- `reports/model_report.txt` – metrici și raport final

## Cum se predă proiectul

1. Creează un repository public pe GitHub.
2. Încarcă toate fișierele din acest folder.
3. Verifică dacă `README.md` este vizibil și clar.
4. Trimite linkul către repository instructorului.
