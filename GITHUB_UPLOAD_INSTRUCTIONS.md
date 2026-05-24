# Instrucțiuni pentru upload pe GitHub

## Varianta 1: direct din browser

1. Mergi la https://github.com/alexandraputanu
2. Apasă **New repository**.
3. Repository name: `product-category-classifier`.
4. Alege **Public**.
5. Nu bifa README, .gitignore sau license. Proiectul le conține deja pe cele necesare.
6. Apasă **Create repository**.
7. Apasă **uploading an existing file**.
8. Încarcă toate folderele și fișierele din proiect.
9. Apasă **Commit changes**.

Link final pentru predare:

```text
https://github.com/alexandraputanu/product-category-classifier
```

## Varianta 2: cu Git din terminal

```bash
cd product-category-classifier
git init
git add .
git commit -m "Initial product category classifier project"
git branch -M main
git remote add origin https://github.com/alexandraputanu/product-category-classifier.git
git push -u origin main
```
