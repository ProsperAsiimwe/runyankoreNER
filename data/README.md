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

✅ Fix 0 Labels: 

sed -i 's/ 0$/ O/g' data/runyankore/train.txt


✅ Remove empty lines: 

sed -i '/^$/d' data/runyankore/train.txt

✅ Verify Labels Again:

cut -d' ' -f2 data/runyankore/train.txt | sort | uniq -c

✅ If still having blank labels, Remove the Last Empty Label

The remaining empty label might be caused by:

A blank space in some lines of train.txt
A trailing space at the end of a word
A line with only whitespace
Run the following command to detect problematic lines:

grep -E " $|^$" data/runyankore/train.txt

✅ If you see empty or improperly formatted lines, clean them with:

sed -i '/^$/d' data/runyankore/train.txt
sed -i 's/ *$//' data/runyankore/train.txt
