#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, contextlib, logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ------------------------ config ------------------------
MODEL_NAME_MAP = {"xlmr": "xlm-roberta-base", "mbert": "bert-base-multilingual-cased"}
DEFAULT_ENTITY_TAGS: Set[str] = {"PER", "LOC", "ORG", "DATE"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"

# ------------------------ logging -----------------------
def init_logger():
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"per_entity_{ts}.log")
    logging.basicConfig(filename=log_file, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

# ------------------------ utils -------------------------
def set_deterministic(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _unit_norm(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / (n + eps)

def _mean_unit(vecs: np.ndarray) -> np.ndarray:
    v = _unit_norm(vecs); return _unit_norm(v.mean(axis=0))

def _safe_cosine(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    if a is None or b is None: return np.nan
    if np.isnan(a).any() or np.isnan(b).any(): return np.nan
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < eps: return np.nan
    return float(np.dot(a, b) / den)

def _read_conll_sentences_loose(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens: yield tokens, tags
                tokens, tags = [], []; continue
            parts = line.split()
            tokens.append(parts[0]); tags.append(parts[-1])
        if tokens: yield tokens, tags

def _coerce_tag(tag: str) -> str:
    if tag == "O": return "O"
    return tag.split("-", 1)[-1] if "-" in tag else tag

def _iter_bio_spans(tokens: List[str], tags: List[str], entity_tags: Set[str]):
    n = len(tokens); i = 0
    while i < n:
        t = tags[i]
        if t == "O": i += 1; continue
        ent = _coerce_tag(t)
        if ent not in entity_tags: i += 1; continue
        j = i + 1
        while j < n:
            tj = tags[j]
            if tj == "O": break
            ej = _coerce_tag(tj)
            pj = tj.split("-", 1)[0] if "-" in tj else "I"
            if ej != ent or pj not in {"I", "B"}: break
            j += 1
        yield i, j, ent; i = j

# ------------------------ extraction --------------------
def extract_embeddings(
    file_path: str, tokenizer, model, layer_indices: List[int],
    entity_tags: Set[str], max_tokens_per_type=500, use_cls=False,
    use_context=False, context_window=2, batch_size=64,
    span_mode="bio", pool="mean", use_amp=False, max_length=128
) -> Tuple[Dict[int, Dict[str, List[np.ndarray]]], Dict[str, int]]:
    assert getattr(tokenizer, "is_fast", False), "Fast tokenizer required (word_ids)."
    per_layer_embeddings = {li: {t: [] for t in entity_tags} for li in layer_indices}
    per_tag_counts = {t: 0 for t in entity_tags}
    fallback_events = 0

    def pool_vecs(sub_vecs: np.ndarray) -> np.ndarray:
        sub_vecs = _unit_norm(sub_vecs)
        if pool == "mean":  return _unit_norm(sub_vecs.mean(axis=0))
        if pool == "max":   return _unit_norm(sub_vecs.max(axis=0))
        if pool == "first": return _unit_norm(sub_vecs[0])
        if pool == "last":  return _unit_norm(sub_vecs[-1])
        return _unit_norm(sub_vecs.mean(axis=0))

    payload = []  # each: {"span_tokens", "rel_idx_list", "tag"}

    def flush():
        nonlocal payload, fallback_events
        if not payload: return
        if use_cls:
            texts = [" ".join(p["span_tokens"]) for p in payload]
            tokd = tokenizer(texts, is_split_into_words=False, return_tensors="pt",
                             truncation=True, max_length=max_length, padding=True).to(DEVICE)
        else:
            spans = [p["span_tokens"] for p in payload]
            tokd = tokenizer(spans, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=max_length, padding=True).to(DEVICE)
        with torch.no_grad():
            cm = (torch.autocast(device_type="cuda",
                                 dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16))
                  if (use_amp and DEVICE == "cuda") else contextlib.nullcontext())
            with cm:
                out = model(**tokd, output_hidden_states=True)
        hs = out.hidden_states  # [emb, layer1..L]

        for li in layer_indices:
            T = hs[li]  # (B,T,H)
            if use_cls:
                embs = _unit_norm(T[:, 0, :].detach().cpu().numpy())
                for i, p in enumerate(payload):
                    tag = p["tag"]
                    if per_tag_counts[tag] >= max_tokens_per_type: continue
                    per_layer_embeddings[li][tag].append(embs[i].astype(np.float32))
                    per_tag_counts[tag] += 1
            else:
                for i, p in enumerate(payload):
                    tag = p["tag"]
                    if per_tag_counts[tag] >= max_tokens_per_type: continue
                    wid = tokd.word_ids(batch_index=i)
                    sub_idx = []
                    for r in p["rel_idx_list"]:
                        sub_idx.extend([j for j, w in enumerate(wid) if w == r])
                    if not sub_idx:
                        first = [j for j, w in enumerate(wid) if w is not None][:1]
                        if not first: continue
                        sub_idx = first; fallback_events += 1
                    sub = T[i, sub_idx, :].detach().cpu().numpy()
                    emb = pool_vecs(sub).astype(np.float32)
                    per_layer_embeddings[li][tag].append(emb)
                    per_tag_counts[tag] += 1
        payload = []

    for toks, tags in tqdm(_read_conll_sentences_loose(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        if span_mode == "bio":
            for s, e, ent in _iter_bio_spans(toks, tags, entity_tags):
                if per_tag_counts[ent] >= max_tokens_per_type: continue
                if use_context:
                    cs = max(0, s - context_window); ce = min(len(toks), e + context_window)
                else:
                    cs, ce = s, e
                span = toks[cs:ce]; 
                rel = list(range(s - cs, e - cs))
                payload.append({"span_tokens": span, "rel_idx_list": rel, "tag": ent})
                if len(payload) >= batch_size: flush()
        else:
            for i, tg in enumerate(tags):
                ent = _coerce_tag(tg)
                if ent not in entity_tags: continue
                if per_tag_counts[ent] >= max_tokens_per_type: continue
                if use_context:
                    cs = max(0, i - context_window); ce = min(len(toks), i + context_window + 1)
                else:
                    cs, ce = i, i + 1
                span = toks[cs:ce]; rel = [i - cs]
                payload.append({"span_tokens": span, "rel_idx_list": rel, "tag": ent})
                if len(payload) >= batch_size: flush()
    flush()
    if fallback_events: logging.warning(f"Tokenizer fallback events: {fallback_events} for {os.path.basename(file_path)}")
    return per_layer_embeddings, per_tag_counts

# ------------------------ main logic --------------------
def per_entity_cosines(language_files: Dict[str, str], model_type: str, layer_indices: List[int],
                       entity_tags: Set[str], out_dir: str, **extract_kwargs):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[model_type])
    mdl = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]).to(DEVICE); mdl.eval()
    os.makedirs(out_dir, exist_ok=True)

    all_by_layer: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {li:{} for li in layer_indices}
    counts_by_lang: Dict[str, Dict[str, int]] = {}

    for lang, path in tqdm(language_files.items(), desc=f"Extract — {model_type}"):
        if not os.path.isfile(path):
            logging.warning(f"[skip] Missing {lang}: {path}"); continue
        per_layer, counts = extract_embeddings(path, tok, mdl, layer_indices, entity_tags, **extract_kwargs)
        counts_by_lang[lang] = counts
        for li in layer_indices: all_by_layer[li][lang] = per_layer[li]

    # compute per-entity cosine tables vs target
    for li in layer_indices:
        rows = {}
        target = all_by_layer[li][TARGET_LANG]
        for lang, tag2vecs in all_by_layer[li].items():
            if lang == TARGET_LANG: continue
            row={}
            for tag in entity_tags:
                tvecs = target[tag]; lvecs = tag2vecs[tag]
                if len(tvecs)==0 or len(lvecs)==0: row[tag]=np.nan; continue
                t_proto = _mean_unit(np.stack(tvecs))
                l_proto = _mean_unit(np.stack(lvecs))
                row[tag] = _safe_cosine(t_proto, l_proto)
            rows[lang]=row
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.to_csv(os.path.join(out_dir, f"per_entity_cosine_to_{TARGET_LANG}_layer{li}.csv"))

    # simple per-tag heatmaps over layers
    tags = sorted(list(entity_tags))
    languages = [l for l in language_files.keys() if l != TARGET_LANG]
    for tag in tags:
        M = np.zeros((len(languages), len(layer_indices))); M[:] = np.nan
        for j, li in enumerate(layer_indices):
            df = pd.read_csv(os.path.join(out_dir, f"per_entity_cosine_to_{TARGET_LANG}_layer{li}.csv"), index_col=0)
            M[:, j] = df[tag].reindex(languages).values
        plt.figure(figsize=(1.8+0.5*len(layer_indices), 0.8+0.35*len(languages)))
        im = plt.imshow(M, aspect="auto", interpolation="nearest"); plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.yticks(range(len(languages)), languages); plt.xticks(range(len(layer_indices)), layer_indices, rotation=45)
        plt.title(f"{model_type} — tag={tag} vs {TARGET_LANG} (cosine)"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_tag_{tag}_{model_type}.png"), dpi=200); plt.close()

def get_all_layers(model_type: str) -> List[int]:
    tmp = AutoModel.from_pretrained(MODEL_NAME_MAP[model_type])
    L = tmp.config.num_hidden_layers; del tmp; return list(range(1, L+1))

def main():
    init_logger()
    ap = argparse.ArgumentParser("Per-entity cross-lingual similarity (cosine)")
    ap.add_argument("--model_type", choices=["xlmr","mbert"], default="xlmr")
    ap.add_argument("--output_dir", default="outputs_per_entity")
    ap.add_argument("--langs", default="all")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--entity_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--use_cls", action="store_true")
    ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--context_window", type=int, default=2)
    ap.add_argument("--max_tokens_per_type", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--span_mode", default="bio", choices=["bio","token"])
    ap.add_argument("--pool", default="mean", choices=["mean","max","first","last"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_deterministic(args.seed)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    language_files = {
        "run": os.path.join(BASE_DIR, "COMBINED/train.txt"),
        "bam": os.path.join(BASE_DIR, "MasakhaNER2.0/bam/train.txt"),
        "bbj": os.path.join(BASE_DIR, "MasakhaNER2.0/bbj/train.txt"),
        "ewe": os.path.join(BASE_DIR, "MasakhaNER2.0/ewe/train.txt"),
        "fon": os.path.join(BASE_DIR, "MasakhaNER2.0/fon/train.txt"),
        "hau": os.path.join(BASE_DIR, "MasakhaNER2.0/hau/train.txt"),
        "ibo": os.path.join(BASE_DIR, "MasakhaNER2.0/ibo/train.txt"),
        "kin": os.path.join(BASE_DIR, "MasakhaNER2.0/kin/train.txt"),
        "lug": os.path.join(BASE_DIR, "MasakhaNER2.0/lug/train.txt"),
        "luo": os.path.join(BASE_DIR, "MasakhaNER2.0/luo/train.txt"),
        "mos": os.path.join(BASE_DIR, "MasakhaNER2.0/mos/train.txt"),
        "nya": os.path.join(BASE_DIR, "MasakhaNER2.0/nya/train.txt"),
        "pcm": os.path.join(BASE_DIR, "MasakhaNER2.0/pcm/train.txt"),
        "sna": os.path.join(BASE_DIR, "MasakhaNER2.0/sna/train.txt"),
        "swa": os.path.join(BASE_DIR, "MasakhaNER2.0/swa/train.txt"),
        "tsn": os.path.join(BASE_DIR, "MasakhaNER2.0/tsn/train.txt"),
        "twi": os.path.join(BASE_DIR, "MasakhaNER2.0/twi/train.txt"),
        "wol": os.path.join(BASE_DIR, "MasakhaNER2.0/wol/train.txt"),
        "xho": os.path.join(BASE_DIR, "MasakhaNER2.0/xho/train.txt"),
        "yor": os.path.join(BASE_DIR, "MasakhaNER2.0/yor/train.txt"),
        "zul": os.path.join(BASE_DIR, "MasakhaNER2.0/zul/train.txt"),
    }
    if args.langs != "all":
        keep = {s.strip() for s in args.langs.split(",") if s.strip()}
        language_files = {k:v for k,v in language_files.items() if k in keep}
    language_files = {k:v for k,v in language_files.items() if os.path.isfile(v)}
    if TARGET_LANG not in language_files:
        logging.error("Target language not present. Abort."); return

    if args.layers == "all":
        layer_indices = get_all_layers(args.model_type)
    else:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]

    entity_tags = {t.strip() for t in args.entity_tags.split(",") if t.strip()}
    out_dir = os.path.join(args.output_dir, args.model_type); os.makedirs(out_dir, exist_ok=True)

    per_entity_cosines(
        language_files, args.model_type, layer_indices, entity_tags, out_dir,
        max_tokens_per_type=args.max_tokens_per_type, use_cls=args.use_cls,
        use_context=args.use_context, context_window=args.context_window,
        batch_size=args.batch_size, span_mode=args.span_mode, pool=args.pool,
        use_amp=args.fp16, max_length=args.max_length
    )

if __name__ == "__main__":
    main()
