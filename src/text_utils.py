import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


EXTRA_STOPWORDS = {
    "please",
    "help",
    "hi",
    "hello",
    "thanks",
    "thank",
}

STOPWORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(tokens)
