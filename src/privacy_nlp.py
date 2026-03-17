"""
Privacy Layer for Data Sanitization.
Intercepts PII (Personally Identifiable Information) before downstream processing.
"""
import re
import pandas as pd
from typing import List
import logging

try:
    import spacy
except ImportError:
    spacy = None

# Initialize logger
logger = logging.getLogger(__name__)

# Global spaCy model to avoid reloading overhead per row
_nlp = None

def get_nlp_model():
    """Lazy-loads the Japanese NER model, fetching it if missing."""
    global _nlp
    if _nlp is None and spacy is not None:
        try:
            _nlp = spacy.load("ja_core_news_sm")
        except OSError:
            logger.warning("Local ja_core_news_sm not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "ja_core_news_sm"], check=True)
            _nlp = spacy.load("ja_core_news_sm")
    return _nlp

def sanitize_text(text: str) -> str:
    """
    Redacts PII from Japanese text using Regex and NER.
    Targets: Emails, Phone Numbers, and PERSON entities.
    
    Note: GPE (Geopolitical Entities) and LOC (Locations) are explicitly 
    ignored to preserve node-based tourism analysis (e.g., Tojinbo, Fukui).
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return text

    clean_text = text

    # 1. Regex: Redact Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    clean_text = re.sub(email_pattern, '[REDACTED_EMAIL]', clean_text)

    # 2. Regex: Redact Japanese Phone Numbers
    # Matches common formats: 090-XXXX-XXXX, 03-XXXX-XXXX, etc.
    phone_pattern = r'\b0\d{1,4}[-(]?\d{1,4}[-(]?\d{3,4}\b'
    clean_text = re.sub(phone_pattern, '[REDACTED_PHONE]', clean_text)

    # 3. NER: Redact Person Names via spaCy
    nlp = get_nlp_model()
    if nlp is not None:
        doc = nlp(clean_text)
        
        # Process in reverse order to avoid index shifting during replacement
        for ent in reversed(doc.ents):
            if ent.label_ == "PERSON":
                # Slice and replace to ensure the exact entity is replaced safely
                clean_text = (
                    clean_text[:ent.start_char] + 
                    '[REDACTED_PERSON]' + 
                    clean_text[ent.end_char:]
                )

    return clean_text

def apply_privacy_layer(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Applies the sanitization function across specified dataframe columns.
    """
    df_clean = df.copy()
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(sanitize_text)
    return df_clean