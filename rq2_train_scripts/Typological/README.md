## Typology Similarity Bar Chart Data Sources

| Script | Bar Chart File | CSV File Containing Plotted Values | Column(s) Used | Description |
|:-------|:----------------|:-----------------------------------|:----------------|:-------------|
| `URIEL/lang2Vec - typology_pipeline.py` | `similarity_to_target_barh.png` | **`cosine_similarity_typology.csv`** | Column for the target language (e.g., `nyn`) | Cosine similarity to Runyankore computed from concatenated URIEL typological feature vectors (syntax, phonology, and phoneme inventory), optionally including genetic and geographic priors. |
| `LinguaMeta - typology_similarity.py` | `fig1_bar_equal.png` | **`typology_similarity_scores.csv`** | `S_total` | Overall typological similarity (S_total) to Runyankore derived from script match, shared country/region locale overlap, and geographic proximity based on latitude–longitude distance. |
