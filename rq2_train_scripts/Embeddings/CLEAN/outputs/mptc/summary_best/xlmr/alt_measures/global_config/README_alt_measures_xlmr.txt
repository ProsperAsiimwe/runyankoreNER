Global-config view (single fixed configuration across all layers):
- Chosen automatically as the (config,tokens,ctx) triple that appears most often as a per-layer winner;
  if tied (or all once), pick the tied triple with the best score_mean_over_langs (highest).
- Selected: c2_t400_c3
- values_by_layer_{tech}_{model}.csv: per-language values using ONLY the global config for every layer.
- lines_{tech}_{model}.png: one line per language; SAME configuration at every layer.
