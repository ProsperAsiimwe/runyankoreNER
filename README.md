# RunyaNER: Named Entity Recognition for Runyankore via Cross-Lingual Transfer

This repository contains the datasets, code, and experimental pipelines used in the MSc thesis:

> **Cross-lingual Adaptation For Named Entity Recognition in Runyankore: 

> Resource Creation, typological, metadata, and Similarity-Driven Auxiliary Language Selection**

The project introduces **RunyaNER**, the first Named Entity Recognition (NER) dataset for **Runyankore**, a low-resource Bantu language spoken in western Uganda, and systematically evaluates **auxiliary language selection strategies** for multilingual transfer.


## Contributions

This repository makes the following research contributions:

- **RunyaNER Dataset**  
  The first publicly available Runyankore NER corpus, annotated under **MasakhaNER 2.0** guidelines with the entity types `PER`, `LOC`, `ORG`, and `DATE`.

- **Semi-Automated Annotation Pipeline**  
  A reproducible human-in-the-loop annotation workflow leveraging **cross-lingual transfer from Luganda** using XLM-RoBERTa, significantly reducing annotation time while preserving quality.

- **Auxiliary Language Selection Framework**  
  Implementation and evaluation of:
  - Typology-based similarity (URIEL)
  - Metadata-based similarity (LinguaMeta)
  - Embedding-based similarity (cosine and Sliced Wasserstein Distance)

- **Reproducible Multilingual NER Experiments**  
  Baselines and transfer experiments using **XLM-R**, **AfroXLM-R**, and **mBERT**, evaluated under monolingual, zero-shot, and cross-lingual settings.


## Dataset Description

### RunyaNER

RunyaNER is constructed from two publicly available parallel corpora:

- **SALT**  
  Derived from the Sunbird African Language Technology (SALT) corpus, containing multi-domain text including news, public communication, and dialogue.

- **MPTC**  
  Extracted from the Multilingual Parallel Text Corpora (MPTC) for East African languages, providing shorter, cleaner sentence pairs.

Together, these sources provide approximately **35,000 Runyankore sentences** prior to filtering.  
Final splits are stratified into **train / dev / test** partitions.

All datasets use the **CoNLL BIO format**, for example:

Abahekyera O
ba O
Allied B-ORG
Democratic I-ORG
Forces I-ORG
bakaba O
nibaza O
kurwanisa O
gavumenti O
ya O
Museveni B-PER
omuri O
Kampala B-LOC
. O


## Annotation Pipeline

Annotation follows a **semi-automated, human-in-the-loop process**:

1. Fine-tune **XLM-R** on **Luganda** NER data from MasakhaNER 2.0
2. Use the model to pre-annotate Runyankore text
3. Import predictions into **Doccano**
4. Manually verify and correct all entity spans
5. Export to CoNLL BIO format

This approach replaces full manual annotation with **verification and boundary correction**, substantially reducing annotation time while maintaining consistency with MasakhaNER guidelines.


## Experimental Setup

### Models

- `xlm-roberta-base`
- `Davlan/afro-xlmr-base`
- `bert-base-multilingual-cased`

### Training Regimes

- **Monolingual fine-tuning** (Runyankore only)
- **Zero-shot transfer** (auxiliary → Runyankore)
- **Cross-lingual co-training** (Runyankore + auxiliaries)

### Evaluation

- Entity-level span **Precision / Recall / F₁**
- MasakhaNER evaluation protocol
- Statistical correlation with similarity metrics


## Auxiliary Language Selection

The repository implements and compares three selection strategies:

1. **Typology-based similarity**  
   Using URIEL/lang2vec vectors

2. **Metadata-based similarity**  
   Using LinguaMeta features (script, geography, locale)

3. **Embedding-based similarity**  
   Using sentence embeddings and:
   - Cosine similarity
   - Sliced Wasserstein Distance (SWD)

Correlation analyses (Pearson’s *r*, Spearman’s *ρ*) quantify how well each similarity measure predicts NER transfer performance.


## License
This project is released under the MIT License.  
All datasets are redistributed under their original licenses, where applicable.  

This work was conducted using resources provided by the University of Cape Town (UCT) under the terms of a non-exclusive license granted to UCT by the author.

## Acknowledgements
This work was supervised by Dr. Jan Buys and Dr. François Meyer from the Department of Computer Science at the University of Cape Town.  

The project builds on and contributes to the MasakhaNER African NLP community and makes use of resources from Sunbird AI, LinguaMeta, URIEL/Lang2Vec, and MasakhaNER 2.0.
