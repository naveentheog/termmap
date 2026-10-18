"""Data layer for the real HiLabs TermMap files.

train csv : entity, chunk_text, coding_system (ICD10CM | SNOMEDCT_US), code, display, subconcept
icd xlsx  : Code, Description            (1-2 descriptions per code: full + abbreviated)
snomed xlsx (sheet 'SNOMED codes'): snomed_code, description, description_order

All codes are kept as strings. Parsed dictionaries are cached as pickles next to the source.
"""
import os
import re

import pandas as pd

ICD, SNO = "ICD10CM", "SNOMEDCT_US"
SYSTEMS = (ICD, SNO)
_TAG = re.compile(r"\s*\(([^()]+)\)\s*$")


def load_train(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df.reset_index(drop=True)


def _cached(path, tag, builder):
    cache = f"{os.path.splitext(path)[0]}.{tag}.pkl"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return pd.read_pickle(cache)
    out = builder()
    out.to_pickle(cache)
    return out


def load_icd(path):
    """-> DataFrame[code, desc, is_leaf, block]; range codes (A00-B99) removed."""
    def build():
        d = pd.read_excel(path, dtype=str).rename(columns={"Code": "code", "Description": "desc"})
        d = d.dropna()
        d["code"], d["desc"] = d.code.str.strip(), d.desc.str.strip()
        d = d[~d.code.str.contains("-")].drop_duplicates()
        codes = set(d.code)
        parents = set()
        for c in codes:
            k = c.replace(".", "")
            for n in range(3, len(k)):
                p = k[:3] + ("." + k[3:n] if n > 3 else "")
                if p in codes:
                    parents.add(p)
        d["is_leaf"] = ~d.code.isin(parents)
        d["block"] = d.code.str[:3]
        return d.reset_index(drop=True)
    return _cached(path, "icd", build)


def load_snomed(path):
    """-> DataFrame[code, desc, tag, order]; tag = semantic tag from the FSN, e.g. 'disorder'.
    The '(tag)' suffix is stripped from the description text used for matching."""
    def build():
        d = pd.read_excel(path, sheet_name="SNOMED codes", dtype=str)
        d = d.rename(columns={"snomed_code": "code", "description": "desc", "description_order": "order"})
        d = d.dropna(subset=["code", "desc"])
        d["code"], d["desc"] = d.code.str.strip(), d.desc.str.strip()
        tags = d.desc.str.extract(_TAG)[0]
        freq = tags.value_counts()
        valid = set(freq[freq >= 200].index)
        ok = tags.isin(valid)
        tag_by_code = tags[ok].groupby(d.code[ok]).first()
        d["tag"] = d.code.map(tag_by_code).fillna("unknown")
        d["desc"] = d.desc.where(~ok, d.desc.str.replace(_TAG, "", regex=True)).str.strip()
        d = d[d.desc != ""].drop_duplicates(["code", "desc"])
        d["order"] = pd.to_numeric(d["order"], errors="coerce").fillna(99).astype(int)
        return d.reset_index(drop=True)
    return _cached(path, "snomed", build)


def dictionaries(icd_path, snomed_path):
    icd, sno = load_icd(icd_path), load_snomed(snomed_path)
    print(f"[data] ICD: {icd.code.nunique():,} codes ({icd[icd.is_leaf].code.nunique():,} leaf), "
          f"{len(icd):,} descriptions")
    print(f"[data] SNOMED: {sno.code.nunique():,} codes, {len(sno):,} descriptions, "
          f"{sno.tag.nunique()} semantic tags")
    return {ICD: icd, SNO: sno}
