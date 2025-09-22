#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, contextlib, logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME_MAP = {"xlmr": "xlm-roberta-base", "mbert": "bert-base-multilingual-cased"}
DEFAULT_ENTITY_TAGS: Set[str] = {"PER","LOC","ORG","DATE"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LANG = "run"

def init_logger():
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"alt_measures_{ts}.log")
    logging.basicConfig(filename=log_file, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    console = logging.StreamHandler(); console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

def set_deterministic(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _unit_norm(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / (n + eps)

def _mean_unit(vecs):
    v = _unit_norm(vecs); return _unit_norm(v.mean(axis=0))

def _safe_cos(a,b,eps=1e-8):
    if a is None or b is None: return np.nan
    if np.isnan(a).any() or np.isnan(b).any(): return np.nan
    den = np.linalg.norm(a)*np.linalg.norm(b)
    if den < eps: return np.nan
    return float(np.dot(a,b)/den)

def _euclid(a,b):
    if a is None or b is None: return np.nan
    if np.isnan(a).any() or np.isnan(b).any(): return np.nan
    return float(np.linalg.norm(a-b))

def _centered_cos(a,b,eps=1e-8):
    # Pearson correlation across dimensions
    a0 = a - a.mean(); b0 = b - b.mean()
    da = np.linalg.norm(a0); db = np.linalg.norm(b0)
    if da < eps or db < eps: return np.nan
    return float(np.dot(a0,b0)/(da*db))

def _linear_cka(X: np.ndarray, Y: np.ndarray, center=True, eps=1e-12) -> float:
    """
    Linear CKA between two reps with same #rows (samples). Here we use tag prototypes as samples.
    X,Y: shape (n_samples, dim)  -- require same n_samples and dim
    """
    if X.shape != Y.shape: return np.nan
    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
    XT_Y = X.T @ Y
    num = np.linalg.norm(XT_Y, ord='fro')**2
    den = np.linalg.norm(X.T @ X, ord='fro') * np.linalg.norm(Y.T @ Y, ord='fro') + eps
    return float(num / den)

def _read_conll_sentences_loose(path: str):
    with open(path, "r", encoding="utf-8") as f:
        toks, tags = [], []
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if toks: yield toks, tags
                toks, tags = [], []; continue
            parts = line.split()
            toks.append(parts[0]); tags.append(parts[-1])
        if toks: yield toks, tags

def _coerce_tag(tag: str) -> str:
    return "O" if tag == "O" else (tag.split("-",1)[-1] if "-" in tag else tag)

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

def extract_embeddings(file_path, tokenizer, model, layer_indices, entity_tags,
                       max_tokens_per_type=500, use_cls=False, use_context=False,
                       context_window=2, batch_size=64, span_mode="bio", pool="mean",
                       use_amp=False, max_length=128):
    assert getattr(tokenizer,"is_fast",False)
    per_layer = {li:{t:[] for t in entity_tags} for li in layer_indices}
    counts = {t:0 for t in entity_tags}
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
                    t=p["tag"]; 
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
                    per_layer[li][t].append(pool_vecs(sub).astype(np.float32)); counts[t]+=1
        payload=[]
    for toks, tags in tqdm(_read_conll_sentences_loose(file_path), desc=f"Reading {os.path.basename(file_path)}"):
        if span_mode=="bio":
            for s,e,ent in _iter_bio_spans(toks,tags,entity_tags):
                if counts[ent] >= max_tokens_per_type: continue
                cs = max(0, s - context_window) if use_context else s
                ce = min(len(toks), e + context_window) if use_context else e
                rel=list(range(s-cs,e-cs))
                payload.append({"span_tokens": toks[cs:ce], "rel_idx_list": rel, "tag": ent})
                if len(payload)>=batch_size: flush()
        else:
            for i,tg in enumerate(tags):
                ent=_coerce_tag(tg)
                if ent not in entity_tags or counts[ent]>=max_tokens_per_type: continue
                cs=max(0, i-context_window) if use_context else i
                ce=min(len(toks), i+context_window+1) if use_context else i+1
                rel=[i-cs]
                payload.append({"span_tokens": toks[cs:ce], "rel_idx_list": rel, "tag": ent})
                if len(payload)>=batch_size: flush()
    flush()
    if fallback: logging.warning(f"Tokenizer fallbacks: {fallback} in {os.path.basename(file_path)}")
    return per_layer, counts

def get_layers(model_type):
    tmp=AutoModel.from_pretrained(MODEL_NAME_MAP[model_type]); L=tmp.config.num_hidden_layers; del tmp
    return list(range(1,L+1))

def main():
    init_logger()
    ap=argparse.ArgumentParser("Alternative measures: Euclidean, centered cosine, CKA on tag prototypes")
    ap.add_argument("--model_type", choices=["xlmr","mbert"], default="xlmr")
    ap.add_argument("--output_dir", default="outputs_alt_measures")
    ap.add_argument("--langs", default="all"); ap.add_argument("--layers", default="all")
    ap.add_argument("--entity_tags", default="PER,LOC,ORG,DATE")
    ap.add_argument("--use_cls", action="store_true"); ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--context_window", type=int, default=2); ap.add_argument("--max_tokens_per_type", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--span_mode", default="bio", choices=["bio","token"])
    ap.add_argument("--pool", default="mean", choices=["mean","max","first","last"])
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    set_deterministic(args.seed)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    language_files = {
        "run": os.path.join(BASE_DIR, "MPTC/train.txt"),
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
        keep={s.strip() for s in args.langs.split(",") if s.strip()}
        language_files={k:v for k,v in language_files.items() if k in keep}
    language_files={k:v for k,v in language_files.items() if os.path.isfile(v)}
    if TARGET_LANG not in language_files: logging.error("Target language not present"); return

    layers = get_layers(args.model_type) if args.layers=="all" else [int(x) for x in args.layers.split(",") if x.strip()]
    tags = [t.strip() for t in args.entity_tags.split(",") if t.strip()]
    out_dir = os.path.join(args.output_dir, args.model_type); os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME_MAP[args.model_type])
    mdl = AutoModel.from_pretrained(MODEL_NAME_MAP[args.model_type]).to(DEVICE); mdl.eval()

    per_layer: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {li:{} for li in layers}
    for lang, path in tqdm(language_files.items(), desc=f"Extract — {args.model_type}"):
        if not os.path.isfile(path): continue
        layer2, _ = extract_embeddings(path, tok, mdl, layers, set(tags),
                                       max_tokens_per_type=args.max_tokens_per_type,
                                       use_cls=args.use_cls, use_context=args.use_context,
                                       context_window=args.context_window, batch_size=args.batch_size,
                                       span_mode=args.span_mode, pool=args.pool,
                                       use_amp=args.fp16, max_length=args.max_length)
        for li in layers: per_layer[li][lang]=layer2[li]

    for li in layers:
        tgt = per_layer[li][TARGET_LANG]
        rows={}
        for lang, tag2embs in per_layer[li].items():
            if lang == TARGET_LANG: continue
            row={}
            # per-tag measures on prototypes
            for tag in tags:
                tv = tgt[tag]; lv = tag2embs[tag]
                if len(tv)==0 or len(lv)==0:
                    row[f"{tag}_cos"]=np.nan; row[f"{tag}_euclid"]=np.nan; row[f"{tag}_ccos"]=np.nan; continue
                tproto=_mean_unit(np.stack(tv)); lproto=_mean_unit(np.stack(lv))
                row[f"{tag}_cos"]=_safe_cos(tproto,lproto)
                row[f"{tag}_euclid"]=_euclid(tproto,lproto)
                row[f"{tag}_ccos"]=_centered_cos(tproto,lproto)
            # CKA over stacked tag prototypes (same sample set = tags)
            try:
                XT = np.stack([_mean_unit(np.stack(tgt[t])) for t in tags])
                YL = np.stack([_mean_unit(np.stack(tag2embs[t])) for t in tags])
                row["CKA_proto_tags"]=_linear_cka(XT,YL,center=True)
            except Exception:
                row["CKA_proto_tags"]=np.nan
            rows[lang]=row
        df=pd.DataFrame.from_dict(rows, orient="index")
        df.to_csv(os.path.join(out_dir, f"alt_measures_to_{TARGET_LANG}_layer{li}.csv"))

if __name__=="__main__":
    main()
