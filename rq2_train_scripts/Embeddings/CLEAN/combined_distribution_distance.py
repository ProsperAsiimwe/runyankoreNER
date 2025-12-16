#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, contextlib, logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt  # NEW

MODEL_NAME_MAP = {
    "afroxlmr": "Davlan/afro-xlmr-base",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}
DEFAULT_ENTITY_TAGS={"PER","LOC","ORG","DATE"}
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG="run"

def init_logger():
    os.makedirs("logs", exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file=os.path.join("logs", f"swd_{ts}.log")
    logging.basicConfig(filename=log_file, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    console=logging.StreamHandler(); console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

def set_deterministic(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _unit_norm(v, eps=1e-8):
    n=np.linalg.norm(v, axis=-1, keepdims=True); return v/(n+eps)

def _read_conll_sentences_loose(path: str):
    with open(path, "r", encoding="utf-8") as f:
        toks, tags=[], []
        for raw in f:
            line=raw.rstrip("\n")
            if not line.strip():
                if toks: yield toks, tags
                toks, tags=[], []; continue
            parts=line.split()
            toks.append(parts[0]); tags.append(parts[-1])
        if toks: yield toks, tags

def _coerce_tag(tag: str): return "O" if tag=="O" else (tag.split("-",1)[-1] if "-" in tag else tag)

def _iter_bio_spans(tokens, tags, entity_tags):
    n=len(tokens); i=0
    while i<n:
        t=tags[i]
        if t=="O": i+=1; continue
        ent=_coerce_tag(t)
        if ent not in entity_tags: i+=1; continue
        j=i+1
        while j<n:
            tj=tags[j]
            if tj=="O": break
            ej=_coerce_tag(tj); pj=tj.split("-",1)[0] if "-" in tj else "I"
            if ej!=ent or pj not in {"I","B"}: break
            j+=1
        yield i,j,ent; i=j

def extract_embs(file_path, tokenizer, model, layer_indices, entity_tags,
                 max_tokens_per_type=1000, use_cls=False, use_context=False,
                 context_window=2, batch_size=64, span_mode="bio", pool="mean",
                 use_amp=False, max_length=128):
    assert getattr(tokenizer,"is_fast",False)
    per_layer={li:{t:[] for t in entity_tags} for li in layer_indices}
    counts={t:0 for t in entity_tags}
    def pool_vecs(sub):
        sub=_unit_norm(sub)
        if pool=="mean":  return _unit_norm(sub.mean(0))
        if pool=="max":   return _unit_norm(sub.max(0))
        if pool=="first": return _unit_norm(sub[0])
        if pool=="last":  return _unit_norm(sub[-1])
        return _unit_norm(sub.mean(0))
    payload=[]; fallback=0
    def flush():
        nonlocal payload,fallback
        if not payload: return
        if use_cls:
            texts=[" ".join(p["span_tokens"]) for p in payload]
            tokd=tokenizer(texts,is_split_into_words=False,return_tensors="pt",truncation=True,max_length=max_length,padding=True).to(DEVICE)
        else:
            spans=[p["span_tokens"] for p in payload]
            tokd=tokenizer(spans,is_split_into_words=True,return_tensors="pt",truncation=True,max_length=max_length,padding=True).to(DEVICE)
        with torch.no_grad():
            cm=(torch.autocast(device_type="cuda",
                dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16))
                if (use_amp and DEVICE=="cuda") else contextlib.nullcontext())
            with cm: out=model(**tokd,output_hidden_states=True)
        hs=out.hidden_states
        for li in layer_indices:
            T=hs[li]
            if use_cls:
                embs=_unit_norm(T[:,0,:].detach().cpu().numpy())
                for i,p in enumerate(payload):
                    t=p["tag"]
                    if counts[t] >= max_tokens_per_type: continue
                    per_layer[li][t].append(embs[i].astype(np.float32)); counts[t]+=1
            else:
                for i,p in enumerate(payload):
                    t=p["tag"]
                    if counts[t] >= max_tokens_per_type: continue
                    wid=tokd.word_ids(batch_index=i)
                    idx=[]
                    for r in p["rel_idx_list"]:
                        idx.extend([j for j,w in enumerate(wid) if w==r])
                    if not idx:
                        first=[j for j,w in enumerate(wid) if w is not None][:1]
                        if not first: continue
                        idx=first; fallback+=1
                    sub=T[i,idx,:].detach().cpu().numpy()
                    per_layer[li][t].append(_unit_norm(pool_vecs(sub)).astype(np.float32)); counts[t]+=1
        payload=[]
    for toks, tg in tqdm(_read_conll_sentences_loose(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        if span_mode=="bio":
            for s,e,ent in _iter_bio_spans(toks,tg,entity_tags):
                if counts[ent] >= max_tokens_per_type: continue

                # Keep your original behavior: no context by default
                cs=max(0,s-ctx) if (ctx:=0) else s
                ce=min(len(toks),e+ctx) if ctx else e

                rel=list(range(s-cs,e-cs))
                payload.append({"span_tokens": toks[cs:ce], "rel_idx_list": rel, "tag": ent})
                if len(payload)>=batch_size: flush()
        else:
            for i,tt in enumerate(tg):
                ent=_coerce_tag(tt)
                if ent not in entity_tags or counts[ent]>=max_tokens_per_type: continue
                payload.append({"span_tokens": [toks[i]], "rel_idx_list": [0], "tag": ent})
                if len(payload)>=batch_size: flush()
    flush()
    return per_layer, counts

def sample_array_list(arr_list: List[np.ndarray], k: int, rng: np.random.Generator) -> np.ndarray:
    if len(arr_list)==0: return np.empty((0,0), dtype=np.float32)
    idx = rng.choice(len(arr_list), size=min(k, len(arr_list)), replace=False)
    X = np.stack([arr_list[i] for i in idx])
    return X

def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, num_proj=256, rng=None) -> float:
    """
    Sliced Wasserstein-1 distance via random projections; assumes X,Y are (n,d) with n>0.
    Uses equal-size samples (truncate to min n).
    """
    if rng is None: rng = np.random.default_rng()
    n = min(len(X), len(Y))
    if n == 0: return np.nan
    X = X[:n]; Y = Y[:n]
    d = X.shape[1]
    total = 0.0
    for _ in range(num_proj):
        w = rng.normal(size=(d,)).astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-8)
        x = X @ w; y = Y @ w
        x.sort(); y.sort()
        total += float(np.mean(np.abs(x - y)))
    return total / num_proj

def get_layers(model_type):
    tmp=AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]); L=tmp.config.num_hidden_layers; del tmp
    return list(range(1, L+1))

# =========================
# NEW: Visualization + summary (mirrors cosine structure)
# =========================

def _save_sorted_bar_swd(df: pd.DataFrame, layer_idx: int, output_dir: str, model_type: str, target_lang: str, top_k_annot: int = 5):
    # For SWD: lower = closer/better → sort ascending
    vals = pd.to_numeric(df["SWD_mean"], errors="coerce").dropna()
    if vals.empty:
        logging.warning(f"Layer {layer_idx}: no valid SWD_mean values; skipping bar plot.")
        return
    order = vals.sort_values(ascending=True)

    plt.figure(figsize=(12, 6))
    plt.bar(order.index, order.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Sliced Wasserstein Distance (mean over tags)")
    plt.title(f"SWD vs {target_lang} — {model_type} — Layer {layer_idx} (lower is closer)")
    for i, (lang, val) in enumerate(order.head(top_k_annot).items()):
        plt.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"swd_vs_{target_lang}_{model_type}_layer{layer_idx}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

def _save_heatmap(language_order, layer_order, mat, output_path: str, title: str):
    plt.figure(figsize=(1.2 + 0.5*len(layer_order), 0.6 + 0.4*len(language_order)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(language_order)), language_order)
    plt.xticks(range(len(layer_order)), layer_order, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def create_visualizations_for_swd_tables(
    swd_tables: Dict[int, pd.DataFrame],
    output_dir: str,
    model_type: str,
    target_lang: str = TARGET_LANG,
):
    os.makedirs(output_dir, exist_ok=True)

    layer_order = sorted(swd_tables.keys())
    ref_layer = layer_order[-1]
    base_df = swd_tables[ref_layer]

    language_order = list(base_df.index)

    mat = np.full((len(language_order), len(layer_order)), np.nan, dtype=float)
    for j, li in enumerate(layer_order):
        df = swd_tables[li].reindex(language_order)
        mat[:, j] = pd.to_numeric(df["SWD_mean"], errors="coerce").values

    heatmap_basename = f"heatmap_swd_layers_{model_type}"
    heatmap_path = os.path.join(output_dir, f"{heatmap_basename}.png")
    _save_heatmap(
        language_order,
        layer_order,
        mat,
        heatmap_path,
        f"Runyankore distance (SWD mean) across layers — {model_type} (lower is closer)"
    )
    pd.DataFrame(mat, index=language_order, columns=layer_order).to_csv(
        os.path.join(output_dir, f"{heatmap_basename}.csv")
    )

    summary_rows = []
    for li in layer_order:
        df = swd_tables[li]
        _save_sorted_bar_swd(df, li, output_dir, model_type=model_type, target_lang=target_lang)

        clean = pd.to_numeric(df["SWD_mean"], errors="coerce").dropna()
        if clean.empty:
            continue

        # Mirror cosine naming: "top5" = best/closest (lowest SWD), "worst5" = largest SWD
        top5 = clean.sort_values(ascending=True).head(5)
        bot5 = clean.sort_values(ascending=False).head(5)

        logging.info(f"=== Layer {li} ({model_type}) — Top 5 (lowest SWD) ===")
        for rank, (lang, val) in enumerate(top5.items(), 1):
            logging.info(f"{rank}. {lang}: swd={val:.6f}")

        logging.info(f"=== Layer {li} ({model_type}) — Worst 5 (highest SWD) ===")
        for rank, (lang, val) in enumerate(bot5.items(), 1):
            logging.info(f"{rank}. {lang}: swd={val:.6f}")

        for lang, val in top5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "top5", "language": lang, "swd_mean": float(val)})
        for lang, val in bot5.items():
            summary_rows.append({"layer": li, "model": model_type, "rank_type": "worst5", "language": lang, "swd_mean": float(val)})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(output_dir, f"summary_top_worst_swd_{model_type}.csv"),
            index=False
        )

def main():
    init_logger()
    ap=argparse.ArgumentParser("Distribution comparison via Sliced Wasserstein Distance (per tag, per layer)")
    ap.add_argument("--model_type", choices=list(MODEL_NAME_MAP.keys()), default="xlmr")
    ap.add_argument("--output_dir", default="outputs_swd")
    ap.add_argument("--langs", default="all"); ap.add_argument("--layers", default="all")
    ap.add_argument("--entity_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--max_tokens_per_type", type=int, default=1000)
    ap.add_argument("--samples_per_tag", type=int, default=200, help="max samples per language/tag used in SWD")
    ap.add_argument("--num_projections", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--span_mode", default="bio", choices=["bio","token"])
    ap.add_argument("--pool", default="mean", choices=["mean","max","first","last"])
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    set_deterministic(args.seed); rng=np.random.default_rng(args.seed)

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
    if args.langs!="all":
        keep={s.strip() for s in args.langs.split(",") if s.strip()}
        language_files={k:v for k,v in language_files.items() if k in keep}
    language_files={k:v for k,v in language_files.items() if os.path.isfile(v)}
    if TARGET_LANG not in language_files:
        logging.error("Target language not present")
        return

    layers=get_layers(args.model_type) if args.layers=="all" else [int(x) for x in args.layers.split(",") if x.strip()]
    tags=[t.strip() for t in args.entity_tags.split(",") if t.strip()]
    out_dir=os.path.join(args.output_dir, args.model_type); os.makedirs(out_dir, exist_ok=True)

    tok=AutoTokenizer.from_pretrained(MODEL_NAME_MAP[args.model_type])
    mdl=AutoModel.from_pretrained(MODEL_NAME_MAP[args.model_type]).to(DEVICE); mdl.eval()

    # collect per-language per-layer embeddings (lists)
    per_layer: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {li:{} for li in layers}
    for lang, path in tqdm(language_files.items(), desc=f"Extract — {args.model_type}"):
        layer2,_ = extract_embs(path, tok, mdl, layers, set(tags),
                                max_tokens_per_type=args.max_tokens_per_type,
                                use_cls=False, use_context=False,
                                context_window=0, batch_size=args.batch_size,
                                span_mode=args.span_mode, pool=args.pool,
                                use_amp=args.fp16, max_length=args.max_length)
        for li in layers: per_layer[li][lang]=layer2[li]

    # compute SWD per tag vs target + keep dfs for reporting
    swd_tables: Dict[int, pd.DataFrame] = {}  # NEW
    for li in layers:
        rows={}
        for lang, tag2 in per_layer[li].items():
            if lang==TARGET_LANG: continue
            row={}
            for tag in tags:
                Xt = sample_array_list(per_layer[li][TARGET_LANG][tag], args.samples_per_tag, rng)
                Xl = sample_array_list(tag2[tag], args.samples_per_tag, rng)
                if Xt.size==0 or Xl.size==0: row[tag]=np.nan; continue
                m=min(len(Xt), len(Xl))
                Xt=Xt[:m]; Xl=Xl[:m]
                row[tag]=sliced_wasserstein(Xt, Xl, num_proj=args.num_projections, rng=rng)
            # aggregate over tags (mean of available)
            vals=[v for v in row.values() if v==v]
            row["SWD_mean"]=float(np.mean(vals)) if vals else np.nan
            rows[lang]=row
        df=pd.DataFrame.from_dict(rows, orient="index")
        df.to_csv(os.path.join(out_dir, f"swd_to_{TARGET_LANG}_layer{li}.csv"))
        swd_tables[li] = df  # NEW

    # NEW: visuals + summary in same run
    if swd_tables:
        create_visualizations_for_swd_tables(
            swd_tables=swd_tables,
            output_dir=out_dir,
            model_type=args.model_type,
            target_lang=TARGET_LANG,
        )

    logging.info("SWD experiment finished successfully.")
    print("\nDone. Outputs in:", out_dir)

if __name__=="__main__":
    main()
