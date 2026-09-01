"""
Text Preprocessing Module for Mental Health NLP Models with Multilingual (Hindi + English) Support.

This module provides text cleaning, Devanagari script preservation, and bilingual psychological
keyword normalization tailored for mental health posts and conversational text statements.
"""

import re
import html
from typing import List, Union, Sequence, Dict
import pandas as pd

CONTRACTIONS_DICT: Dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

# Bilingual psychological mapping for Hindi terms to ensure accurate TF-IDF overlap
HINDI_PSYCH_MAP: Dict[str, str] = {
    "उदासी": "sadness depression",
    "उदास": "sad depressed",
    "अकेलापन": "loneliness alone",
    "अकेला": "lonely alone",
    "चिंता": "anxiety worry",
    "घबराहट": "panic anxiety",
    "डर": "fear panic",
    "तनाव": "stress pressure",
    "थकान": "exhausted fatigue",
    "थका": "tired exhausted",
    "नींद": "sleep insomnia",
    "बेचैनी": "restless anxiety",
    "मरने": "suicidal death",
    "आत्महत्या": "suicide suicidal",
    "परेशान": "distressed overwhelmed",
    "खुश": "happy normal good",
    "संतुष्ट": "satisfied peaceful normal",
    "अच्छा": "good fine normal",
    "शान्त": "calm normal relaxed",
    "उत्साहित": "excited manic euphoric",
    "गुस्सा": "anger dysregulation",
    "खालीपन": "empty hollow void"
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS_DICT.items():
        text = text.replace(contraction, expansion)
    return text


def normalize_bilingual_terms(text: str) -> str:
    """Map key Hindi emotional keywords to psychological synonyms while preserving context."""
    for hindi_term, english_synonym in HINDI_PSYCH_MAP.items():
        if hindi_term in text:
            # Append synonym for semantic feature extraction
            text = text.replace(hindi_term, f"{hindi_term} {english_synonym}")
    return text


def clean_text(text: Union[str, float, None], lower: bool = True) -> str:
    """
    Clean and normalize a single text statement, supporting English and Devanagari (Hindi) script.
    
    Args:
        text: Raw text string (or NaN/None).
        lower: Whether to convert text to lowercase.
        
    Returns:
        Cleaned text string.
    """
    if text is None or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove User Mentions (@user) and Subreddit tags (r/...)
    text = re.sub(r'@\w+|r/\w+|u/\w+', ' ', text)
    
    # Expand English contractions
    text = expand_contractions(text)
    
    # Normalize bilingual Hindi mental health terms
    text = normalize_bilingual_terms(text)
    
    # Convert to lowercase if requested
    if lower:
        text = text.lower()
        
    # Replace non-word/non-alphanumeric characters, while preserving Devanagari Hindi Unicode range (\u0900-\u097F)
    text = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', ' ', text)
    
    # Normalize multiple whitespace characters into single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_corpus(corpus: Sequence[str], lower: bool = True) -> List[str]:
    """
    Clean an iterable/series of text statements.
    
    Args:
        corpus: Iterable or pandas Series of text entries.
        lower: Whether to convert text to lowercase.
        
    Returns:
        List of cleaned text strings.
    """
    return [clean_text(doc, lower=lower) for doc in corpus]
