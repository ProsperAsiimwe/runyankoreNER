'''
python3 pull_from_linguameta.py \
  --languages_csv languages_template.csv \
  --out_features lingua_features.csv

'''

#!/usr/bin/env python3
import argparse, csv, json, math, sys
from typing import Dict, Any, List, Tuple
import requests

LM_BASE = "https://raw.githubusercontent.com/google-research/url-nlp/main/linguameta/data"
LOCALES_URL = f"{LM_BASE}/locales.json"

# Map common ISO 639-3 -> BCP-47 2-letter (and a few macro/edge cases) for your set
ALIASES: Dict[str, List[str]] = {
    # 3-letter -> 2-letter (BCP-47) where applicable
    "bam": ["bm"],       # Bambara
    "ewe": ["ee"],       # Ewe
    "hau": ["ha"],       # Hausa
    "ibo": ["ig"],       # Igbo
    "kin": ["rw"],       # Kinyarwanda
    "lug": ["lg"],       # Ganda/Luganda
    "nya": ["ny"],       # Nyanja/Chichewa
    "sna": ["sn"],       # Shona
    "swa": ["sw"],       # Swahili (macrolanguage)
    "tsn": ["tn"],       # Tswana
    "twi": ["tw", "ak"], # Twi (try 'tw', then Akan 'ak' if needed)
    "wol": ["wo"],       # Wolof
    "xho": ["xh"],       # Xhosa
    "yor": ["yo"],       # Yoruba
    "zul": ["zu"],       # Zulu
    # Codes that already match BCP-47 JSON names (keep empty to try as-is):
    "nyn": [], "bbj": [], "fon": [], "luo": [], "mos": [], "pcm": []
}

ENDANGER_MAP = {
    "SAFE": "safe",
    "VULNERABLE": "vulnerable",
    "DEFINITE": "definitely_endangered",
    "SEVERE": "severely_endangered",
    "CRITICAL": "critically_endangered",
    "EXTINCT": "extinct"
}

def fetch_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def load_locales() -> Dict[str, Dict[str, Any]]:
    data = fetch_json(LOCALES_URL)
    mapping = {}
    for item in data.get("locale_map", []):
        loc = item.get("locale", {})
        code = (loc.get("locale_code") or "").upper()
        if code:
            mapping[code] = {
                "name": loc.get("locale_name"),
                "region": item.get("region"),
                "subregion": item.get("subregion"),
                "group": item.get("regional_group"),
            }
    return mapping

def most_common(items: List[str]) -> str:
    if not items: return ""
    from collections import Counter
    return Counter(items).most_common(1)[0][0]

def choose_primary_script(entries: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    # Prefer canonical; else widespread/official; else any observed
    canonical = []
    widespread_or_official = []
    others = []
    for e in entries:
        sc = e.get("script", {})
        code = sc.get("iso_15924_code")
        if not code: continue
        if sc.get("is_canonical"):
            canonical.append(code)
        elif sc.get("is_in_widespread_use") or sc.get("has_official_status"):
            widespread_or_official.append(code)
        else:
            others.append(code)
    if canonical:
        primary = most_common(canonical)
    elif widespread_or_official:
        primary = most_common(widespread_or_official)
    elif others:
        primary = most_common(others)
    else:
        primary = ""
    alt = sorted({c for c in canonical + widespread_or_official + others if c and c != primary})
    return primary, alt

def aggregate_countries(entries: List[Dict[str, Any]], locales_map: Dict[str, Dict[str, Any]]) -> List[str]:
    codes = set()
    for e in entries:
        loc = e.get("locale", {})
        code = (loc.get("iso_3166_code") or "").upper()
        if code and code != "XXXX":
            codes.add(code)
    names = []
    for c in sorted(codes):
        name = locales_map.get(c, {}).get("name") or c
        names.append(name)
    return names

def aggregate_geolocation(entries: List[Dict[str, Any]]) -> Tuple[float, float]:
    lats, lons = [], []
    for e in entries:
        geo = e.get("geolocation", {})
        lat = geo.get("latitude"); lon = geo.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            lats.append(lat); lons.append(lon)
    if not lats:
        return (float("nan"), float("nan"))
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def aggregate_official_status(entries: List[Dict[str, Any]]) -> str:
    # none < regional < de_facto < official
    best = 0
    for e in entries:
        off = e.get("official_status", {})
        if off.get("has_official_status"): best = max(best, 3)
        if off.get("has_de_facto_official_status"): best = max(best, 2)
        if off.get("has_regional_official_status"): best = max(best, 1)
    return ["none","regional","de_facto","official"][best]

def get_total_population(lang: Dict[str, Any]) -> int:
    tot = lang.get("total_population")
    if isinstance(tot, int):
        return tot
    s = 0
    for e in lang.get("language_script_locale", []):
        sp = e.get("speaker_data", {})
        n = sp.get("number_of_speakers")
        if isinstance(n, int):
            s += n
    return s if s > 0 else ""

def get_endangerment(lang: Dict[str, Any]) -> str:
    end = lang.get("endangerment_status", {}).get("endangerment")
    if not end: return ""
    return ENDANGER_MAP.get(str(end).upper(), "").lower()

def try_fetch_code(fetch_code: str) -> Dict[str, Any]:
    url = f"{LM_BASE}/{fetch_code.lower()}.json"
    return fetch_json(url)

def resolve_and_fetch(original_code: str) -> Tuple[str, Dict[str, Any]]:
    """
    Try the original code; if 404, try known aliases.
    Returns (used_fetch_code, json_dict).
    Raises the last HTTPError if all attempts fail.
    """
    # Try original first
    try:
        data = try_fetch_code(original_code)
        return original_code, data
    except requests.HTTPError as e_first:
        # Try aliases
        for alias in ALIASES.get(original_code.lower(), []):
            try:
                data = try_fetch_code(alias)
                # Found via alias
                print(f"[ok alias={alias}] {original_code}")
                return alias, data
            except requests.HTTPError:
                continue
        # If we had no aliases or all failed, re-raise the first error
        raise e_first

def process_language(original_code: str, locales_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fetch_code, lang = resolve_and_fetch(original_code)
    entries = lang.get("language_script_locale", [])
    script_primary, script_alts = choose_primary_script(entries)
    countries = aggregate_countries(entries, locales_map)
    lat, lon = aggregate_geolocation(entries)
    official = aggregate_official_status(entries)
    speakers = get_total_population(lang)
    endang = get_endangerment(lang)
    return {
        # Keep the *original* code so it stays aligned with your CSVs
        "code": original_code,
        "countries": "; ".join(countries),
        "lat": lat if isinstance(lat, (int,float)) else "",
        "lon": lon if isinstance(lon, (int,float)) else "",
        "script_primary": script_primary or "",
        "script_alt": "; ".join(script_alts),
        "official_status": official,
        "speakers_total": speakers,
        "endangerment": endang or "safe"
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages_csv", required=True, help="CSV with columns: code,name,target")
    ap.add_argument("--out_features", required=True, help="Output CSV (lingua_features.csv)")
    args = ap.parse_args()

    # Load language codes
    codes = []
    with open(args.languages_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            if code:
                codes.append(code)

    locales_map = load_locales()
    out_rows = []
    for code in codes:
        try:
            out_rows.append(process_language(code, locales_map))
            if code.lower() not in ALIASES or not ALIASES[code.lower()]:
                print(f"[ok] {code}")
        except requests.HTTPError as e:
            print(f"[warn] {code}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] {code}: {e}", file=sys.stderr)

    fields = ["code","countries","lat","lon","script_primary","script_alt","official_status","speakers_total","endangerment"]
    with open(args.out_features, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

if __name__ == "__main__":
    main()

