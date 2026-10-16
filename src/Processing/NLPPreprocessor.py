"""NLP preprocessing for scientific sections."""

from __future__ import annotations

import re
import unicodedata

from collections.abc import Iterable


DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all",
    "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "may", "me", "might", "more", "most", "must",
    "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}


class NLPPreprocessor:
    """Convert cleaned scientific prose into a token sequence."""

    TOKEN_PATTERN = re.compile(
        r"[^\W_]+(?:[-'][^\W_]+)*",
        flags=re.UNICODE,
    )

    def __init__(
        self,
        stopwords: Iterable[str] | None = None,
        min_token_length: int = 2,
        remove_stopwords: bool = True,
        preserve_numbers: bool = False,
        spacy_model: str | None = None,
    ) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1.")

        self.min_token_length = min_token_length
        self.remove_stopwords = remove_stopwords
        self.preserve_numbers = preserve_numbers

        self.stopwords = {
            word.casefold()
            for word in (
                DEFAULT_STOPWORDS if stopwords is None else stopwords
            )
        }

        self._nlp = None

        if spacy_model is not None:
            try:
                import spacy
                self._nlp = spacy.load(spacy_model)
            except ImportError as exc:
                raise RuntimeError(
                    "spaCy is required when spacy_model is specified. "
                    "Install it with `pip install spacy`."
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Could not load spaCy model {spacy_model!r}. "
                    "Install the model before enabling lemmatization."
                ) from exc

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize Unicode and lowercase text."""
        return unicodedata.normalize("NFKC", text).casefold()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text while preserving common scientific compounds."""
        text = self.normalize(text)
        return self.TOKEN_PATTERN.findall(text)

    def preprocess(self, text: str) -> list[str]:
        """Return normalized tokens suitable for topic modelling."""
        tokens = self.tokenize(text)
        filtered: list[str] = []

        for token in tokens:
            # Remove numeric-only tokens by default, but retain terms
            # such as "3d" and "covid-19".
            if not self.preserve_numbers and token.isnumeric():
                continue

            if len(token) < self.min_token_length:
                continue

            if self.remove_stopwords and token in self.stopwords:
                continue

            filtered.append(token)

        if self._nlp is not None and filtered:
            # Lemmatize each token while retaining its scientific spelling
            # when spaCy cannot provide a useful lemma.
            doc = self._nlp(" ".join(filtered))
            filtered = [
                token.lemma_.casefold()
                if token.lemma_ and token.lemma_ != "-PRON-"
                else token.text.casefold()
                for token in doc
                if not token.is_space
            ]

        return filtered