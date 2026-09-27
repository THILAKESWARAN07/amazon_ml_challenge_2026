"""
Normalization module for Business Entity Resolution.
Amazon ML Challenge 2026 - Phase 2 & Phase 3

Provides high-performance, non-destructive text, business name, and address normalization
supporting open-set countries and multi-lingual/special characters.
"""

import re
import unicodedata
from typing import Optional, Set, Tuple, List
import pandas as pd


# Legal entity suffixes to identify / standardize
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l l c", "l.l.c.", "ltd", "limited",
    "corp", "corporation", "co", "company", "pvt", "private", "pvt ltd",
    "gmbh", "sarl", "sa", "sas", "llp", "l.l.p.", "pllc", "plc",
    "enterprise", "enterprises", "group", "holdings", "services"
}

# Domain extensions commonly appearing in business names
DOMAIN_EXTENSIONS = (
    ".com", ".org", ".net", ".in", ".co.in", ".co", ".io", ".fr",
    ".de", ".uk", ".ai", ".biz", ".info", ".us", ".store", ".online"
)

# Common address abbreviations
STREET_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters, accents, and umlauts to ASCII equivalents."""
    if not text:
        return ""
    # Normalize unicode to decomposed form and remove non-spacing marks (accents)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ASCII", "ignore").decode("ASCII")
    return ascii_text if ascii_text else text


def normalize_text(text: Optional[str]) -> str:
    """
    Standardize text: lowercase, unicode normalisation, punctuation replacement, whitespace strip.
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    # Unicode normalize
    text = normalize_unicode(text)
    # Lowercase
    text = text.lower()
    # Replace & with and, @ with at
    text = text.replace("&", " and ").replace("@", " at ")
    # Replace non-alphanumeric characters with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_domain_suffix(text: str) -> str:
    """Strip web domain suffixes if present (e.g. google.com -> google)."""
    t = text.strip()
    for ext in DOMAIN_EXTENSIONS:
        if t.endswith(ext):
            t = t[:-len(ext)]
            break
    return t


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize business name:
    - Strip domain suffixes (.com, .org, etc.)
    - Standardize legal suffixes
    - Remove extra punctuation and whitespace
    """
    if name is None or pd.isna(name):
        return ""
    
    cleaned = normalize_unicode(str(name)).lower().strip()
    cleaned = strip_domain_suffix(cleaned)
    cleaned = cleaned.replace("&", " and ").replace("@", " at ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    
    # Filter/standardize trailing legal tokens
    filtered_tokens = []
    for t in tokens:
        if t in {"and", "the", "of", "for", "in", "at", "by", "a", "an"}:
            continue
        filtered_tokens.append(t)
        
    return " ".join(filtered_tokens) if filtered_tokens else " ".join(tokens)


def compact_string(text: Optional[str]) -> str:
    """
    Produce a compact string with all non-alphanumeric characters removed.
    Useful for exact compact matching robust to spacing & punctuation.
    """
    if text is None or pd.isna(text):
        return ""
    text = normalize_unicode(str(text)).lower()
    text = strip_domain_suffix(text)
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize address fields:
    - Standardize street types and directional abbreviations
    - Preserve all house/building numbers, postal/PIN codes
    - Remove punctuation and collapse whitespace
    """
    if address is None or pd.isna(address):
        return ""
    
    cleaned = normalize_unicode(str(address)).lower()
    cleaned = cleaned.replace("&", " and ").replace("#", " number ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    
    tokens = cleaned.split()
    normalized_tokens = [STREET_ABBREVIATIONS.get(t, t) for t in tokens if t]
    return " ".join(normalized_tokens)


def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """
    Extract numbers (house numbers, postal codes, unit numbers) from address/text.
    """
    if text is None or pd.isna(text):
        return []
    return re.findall(r"\b\d+\b", str(text))


def normalize_country(country: Optional[str]) -> str:
    """
    Standardize country string while preserving open-set countries.
    """
    if country is None or pd.isna(country):
        return "UNKNOWN"
    c = str(country).strip().upper()
    return c if c else "UNKNOWN"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full normalization pipeline on a dataset DataFrame, returning a new DataFrame
    with original fields preserved and normalized columns added.
    """
    df_norm = df.copy()
    
    if "business_name" in df_norm.columns:
        df_norm["norm_name"] = df_norm["business_name"].apply(normalize_business_name)
        df_norm["compact_name"] = df_norm["business_name"].apply(compact_string)
        
    if "business_address" in df_norm.columns:
        df_norm["norm_address"] = df_norm["business_address"].apply(normalize_address)
        df_norm["compact_address"] = df_norm["business_address"].apply(compact_string)
        
    if "country" in df_norm.columns:
        df_norm["norm_country"] = df_norm["country"].apply(normalize_country)
        
    return df_norm
