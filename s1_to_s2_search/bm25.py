"""BM25 helpers for S1 -> S2 name/address retrieval.

Stdlib + pandas/numpy only. Implements Okapi BM25 with field weighting
(name x3 + address x1) over word tokens + char trigrams (typo robust).
"""

import re
import unicodedata
from collections import Counter
from math import log

# Abbreviation / legal-suffix normalisation (applied as whole-word replace).
ABBREV = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "pvt.": "private",
    "co": "company",
    "org": "organisation",
    "dept": "department",
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "ste": "suite",
    "apt": "apartment",
    "sarl": "sarl",  # keep French suffixes, just normalise punctuation
    "sa": "sa",
    "sas": "sas",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Words that are too common to block on alone (still scored, just not used for blocking).
STOPWORDS = {
    "the", "and", "of", "inc", "incorporated", "corp", "corporation",
    "ltd", "limited", "pvt", "private", "co", "company", "llc", "llp",
}


def normalize(text: str) -> str:
    """Lowercase, de-accent latin chars, expand abbreviations, strip punct."""
    if text is None:
        return ""
    text = str(text)
    # NFKD + strip combining marks only for latin range (keeps Devanagari etc.)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    words = []
    for w in text.split():
        words.append(ABBREV.get(w, w))
    text = " ".join(words)
    text = _WS_RE.sub(" ", text).strip()
    return text


def word_tokens(norm_text: str):
    return norm_text.split() if norm_text else []


def char_trigrams(norm_text: str):
    """Char 3-grams of the spaceless string. Empty for very short strings."""
    s = norm_text.replace(" ", "")
    if len(s) < 4:
        return []
    return [s[i:i + 3] for i in range(len(s) - 2)]


def doc_tokens(name: str, address: str):
    """Combined tokens used for BM25: words + char trigrams."""
    n = normalize(name)
    a = normalize(address)
    toks = word_tokens(n) + word_tokens(a) + char_trigrams(n) + char_trigrams(a)
    return toks, n, a


def query_tokens(name: str, address: str):
    toks, n, a = doc_tokens(name, address)
    return toks, n, a


def blocking_terms(norm_name: str, norm_addr: str, max_trigrams: int = 30):
    """Terms for the coarse blocking regex: long words + char trigrams.

    Returns (regex_pattern, trigram_list, word_list).
    """
    words = [w for w in word_tokens(norm_name + " " + norm_addr)
             if len(w) >= 4 and w not in STOPWORDS]
    # Prefer name words first
    name_words = [w for w in word_tokens(norm_name) if len(w) >= 4 and w not in STOPWORDS]
    addr_words = [w for w in words if w not in name_words]
    keep_words = (name_words[:8] + addr_words[:4])[:10]

    tris = char_trigrams(norm_name)[:max_trigrams]
    # If name is tiny, back off to address trigrams
    if len(tris) < 5:
        tris = (tris + char_trigrams(norm_addr))[:max_trigrams]

    alts = [re.escape(t) for t in keep_words + tris if t]
    # de-dupe preserving order
    seen, uniq = set(), []
    for a in alts:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    pattern = "|".join(uniq) if uniq else None
    return pattern, tris, keep_words


def compute_idf(docs_tokens, n_docs: int):
    """IDF dict from a list of token lists (local/global DF)."""
    df = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    idf = {}
    for t, f in df.items():
        # Okapi BM25 IDF with +1 floor to avoid negatives
        idf[t] = log(1 + (n_docs - f + 0.5) / (f + 0.5))
    return idf


def bm25_one(query_counts: Counter, doc_counts: Counter, doc_len: int,
             avgdl: float, idf: dict, k1: float = 1.2, b: float = 0.75) -> float:
    denom_norm = 1 - b + b * (doc_len / avgdl) if avgdl else 1.0
    score = 0.0
    for t in query_counts:
        tf = doc_counts.get(t, 0)
        if tf == 0:
            continue
        score += idf.get(t, 0.0) * (tf * (k1 + 1)) / (tf + k1 * denom_norm)
    return score


def bm25_field_weighted(q_name_toks, q_addr_toks, d_name_toks, d_addr_toks,
                        d_name_len, d_addr_len, avg_name_len, avg_addr_len,
                        idf_name, idf_addr, w_name=3.0, w_addr=1.0,
                        k1=1.2, b=0.75):
    """BM25F-lite: weighted sum of name-field and address-field BM25."""
    from collections import Counter as C
    qn, qa = C(q_name_toks), C(q_addr_toks)
    dn, da = C(d_name_toks), C(d_addr_toks)
    s_name = bm25_one(qn, dn, d_name_len, avg_name_len, idf_name, k1, b)
    s_addr = bm25_one(qa, da, d_addr_len, avg_addr_len, idf_addr, k1, b)
    return w_name * s_name + w_addr * s_addr
