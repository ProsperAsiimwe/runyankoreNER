#!/usr/bin/env python3
"""
Combine MasakhaNER2.0 train/dev from a chosen list and pair with Runyankore test sets.

Edit these two variables:
  - OUTPUT_DIR: final output directory to create
  - MASAKHA_LANGUAGES: languages to combine (order preserved)

Layout produced:
  OUTPUT_DIR/
    (combined.train.txt, combined.dev.txt)   # created temporarily, optionally deleted
    SALT/train.txt, dev.txt, test.txt
    MPTC/train.txt, dev.txt, test.txt
    COMBINED/train.txt, dev.txt, test.txt
"""

from pathlib import Path
import shutil
import json
from datetime import datetime

# === EDIT THESE === #
OUTPUT_DIR = Path("../EmbeddingGroups/ZERO-SHOT/twi_luo_kin")  # <<< change to your desired output folder
MASAKHA_LANGUAGES = ["twi", "luo", "kin"]               # <<< change languages here (order matters)
# ================== #

# You can tweak these if your repo layout differs
BASE_DIR = Path(__file__).resolve().parent.parent        # repo root (contains MasakhaNER2.0, SALT, MPTC, COMBINED)
MASAKHA_DIR = BASE_DIR / "MasakhaNER2.0"
# RUNYANKORE_SOURCES = ["SALT", "MPTC", "COMBINED"]
RUNYANKORE_SOURCES = ["COMBINED"]
ENCODING = "utf-8"
FORCE_OVERWRITE = False      # set True to replace an existing OUTPUT_DIR
SKIP_MISSING_LANGS = False   # set True to skip languages missing train/dev

# New: automatically delete temporary combined files after copying to all sources
DELETE_COMBINED_AFTER_USE = True

def verify_exists(p: Path, what: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")
    return p

def concat_text_files(sources, dest: Path, encoding=ENCODING) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dest.open("w", encoding=encoding, newline="") as w:
        for src in sources:
            verify_exists(src, "source file")
            last_had_newline = True
            with src.open("r", encoding=encoding, errors="replace", newline="") as r:
                for line in r:
                    w.write(line)
                    total += 1
                    last_had_newline = line.endswith("\n") or line.endswith("\r")
            if not last_had_newline:
                w.write("\n")
                total += 1
    return total

def count_lines(p: Path, encoding=ENCODING) -> int:
    n = 0
    with p.open("r", encoding=encoding, errors="replace", newline="") as f:
        for _ in f:
            n += 1
    return n

def safe_delete(p: Path):
    try:
        p.unlink()
        print(f"[i] Deleted temporary file: {p}")
    except FileNotFoundError:
        pass

def main():
    # Prepare language sources
    train_srcs, dev_srcs, missing = [], [], []
    for lang in MASAKHA_LANGUAGES:
        lang_dir = MASAKHA_DIR / lang
        train = lang_dir / "train.txt"
        dev = lang_dir / "dev.txt"
        if train.exists() and dev.exists():
            train_srcs.append(train)
            dev_srcs.append(dev)
        else:
            missing.append(lang)

    if missing and not SKIP_MISSING_LANGS:
        raise SystemExit(f"Missing train/dev for languages: {', '.join(missing)} "
                         f"(set SKIP_MISSING_LANGS=True to skip)")
    if missing and SKIP_MISSING_LANGS:
        print(f"[!] Skipping languages with missing files: {', '.join(missing)}")

    if not train_srcs or not dev_srcs:
        raise SystemExit("No valid language files to combine.")

    # Handle output directory
    if OUTPUT_DIR.exists():
        if not FORCE_OVERWRITE:
            raise SystemExit(f"Output exists: {OUTPUT_DIR} (set FORCE_OVERWRITE=True to overwrite)")
        print(f"[!] Removing existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build combined files once
    combined_train = OUTPUT_DIR / "combined.train.txt"
    combined_dev = OUTPUT_DIR / "combined.dev.txt"

    print("[*] Combining train files…")
    train_lines = concat_text_files(train_srcs, combined_train)
    print(f"[✓] Combined train -> {combined_train} ({train_lines} lines)")

    print("[*] Combining dev files…")
    dev_lines = concat_text_files(dev_srcs, combined_dev)
    print(f"[✓] Combined dev   -> {combined_dev} ({dev_lines} lines)")

    # For each Runyankore source, copy combined train/dev + unaltered test.txt
    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "base_dir": str(BASE_DIR),
        "masakha_dir": str(MASAKHA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "languages": MASAKHA_LANGUAGES,
        "skipped_languages": missing if SKIP_MISSING_LANGS else [],
        "sources": {}
    }

    # Track whether every source was written successfully (for safe deletion)
    all_copies_ok = True

    for src in RUNYANKORE_SOURCES:
        test_src = verify_exists(BASE_DIR / src / "test.txt", f"{src} test.txt")
        tgt_dir = OUTPUT_DIR / src
        tgt_dir.mkdir(parents=True, exist_ok=True)

        # Copy combined files
        shutil.copy2(combined_train, tgt_dir / "train.txt")
        shutil.copy2(combined_dev, tgt_dir / "dev.txt")
        shutil.copy2(test_src, tgt_dir / "test.txt")

        # Quick sanity: ensure copied files exist
        ok = (tgt_dir / "train.txt").exists() and (tgt_dir / "dev.txt").exists() and (tgt_dir / "test.txt").exists()
        all_copies_ok = all_copies_ok and ok

        test_count = count_lines(tgt_dir / "test.txt")
        manifest["sources"][src] = {
            "folder": str(tgt_dir),
            "train_lines": train_lines,
            "dev_lines": dev_lines,
            "test_lines": test_count
        }
        print(f"[✓] Built: {tgt_dir}")

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    # Optional cleanup of intermediate combined files
    if DELETE_COMBINED_AFTER_USE:
        if all_copies_ok:
            safe_delete(combined_train)
            safe_delete(combined_dev)
        else:
            print("[!] Skipped deleting combined files because some copies may have failed.")

    print("\nDone.")
    print(f"Top-level output: {OUTPUT_DIR}")
    for src in RUNYANKORE_SOURCES:
        print(f"  - {OUTPUT_DIR / src}")

if __name__ == "__main__":
    main()
