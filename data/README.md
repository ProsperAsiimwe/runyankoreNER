# Runyankore-Dataset

This repository contains a **Named Entity Recognition (NER) dataset** for Runyankore in **CoNLL format**.

## 📌 Dataset Overview

The dataset is structured for **Named Entity Recognition (NER)**, following the **CoNLL-2003 format**. Each token in a sentence is annotated with its corresponding NER label.


### **Example**
John    B-PER
Doe     I-PER
works   O
at      O
Google  B-ORG
.       O

📌 **Explanation:**
- `John` (B-PER) and `Doe` (I-PER) → **Person entity**
- `Google` (B-ORG) → **Organization entity**
- `O` → Outside any named entity

---

pip install --target=/scratch/prosper/python-packages -r requirements.txt

pip install --target=/scratch/prosper/python-packages torch

✅ Delete the Cached Features File:
rm -rf data/runyankore/cached_*


✅ Verify Labels:
cut -d' ' -f2 data/runyankore/train.txt | sort | uniq -c

✅ Reload .env and Verify Paths
source .env
echo $DATA_DIR
echo $OUTPUT_DIR




