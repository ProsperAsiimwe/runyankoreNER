Per-layer 'upper envelope' view:
- best_config_by_layer_{tech}_{model}.csv: winner per layer with tokens, ctx_window, mean score.
- values_by_layer_{tech}_{model}.csv: per-language values; each column uses the winner for that layer.
- values_with_config_by_layer_{tech}_{model}.csv: merged table (layer + winner + per-language values).
- lines_{tech}_{model}.png: one line per language; labels above x-axis show the winner for each layer.
