import re
import string
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

__all__ = ["Preprocessor"]

# Ensure NLTK resources are available
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class Preprocessor:
    """
    A class for text preprocessing that includes language detection,
    stop-word removal, and stemming.
    """

    _jp_regex = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", re.UNICODE)
    _stop_words = set(stopwords.words("english"))
    _stemmer = PorterStemmer()

    @classmethod
    def preprocess_text(cls, text: str) -> str:
        """
        Lowercase, strip punctuation, tokenize, remove non-alpha tokens
        and English stopwords, then stem each token.

        Args:
            text: The raw text string.
        Returns:
            A space-joined string of processed/stemmed tokens.
        """
        text = text.lower()
        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # tokenize into words
        tokens = word_tokenize(text)
        # filter and stem
        filtered = [
            cls._stemmer.stem(tok)
            for tok in tokens
            if tok.isalpha() and tok not in cls._stop_words
        ]
        return " ".join(filtered)

    @classmethod
    def is_english(cls, text: str) -> bool:
        """
        Detect whether a text is English:
        1) Quick regex check for Japanese characters
        2) Fallback to langdetect

        Args:
            text: The text to check.
        Returns:
            True if likely English, False otherwise.
        """
        if cls._jp_regex.search(text):
            return False
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    @classmethod
    def preprocess_record(cls, record: dict, fields=None) -> dict:
        """
        Preprocess specified text fields in a single record.

        Args:
            record: A dict containing text fields.
            fields: An iterable of field names to preprocess."""
        if fields is None:
            fields = [k for k, v in record.items() if isinstance(v, str)]
        new = record.copy()
        for field in fields:
            if field in new and isinstance(new[field], str):
                new[field] = cls.preprocess_text(new[field])
        return new

    @classmethod
    def preprocess_records(
        cls, records: list, fields=None, filter_non_english=True
    ) -> list:
        """
        Process a list of records (dicts), optionally filtering by English.

        Args:
            records: List of dicts to process.
            fields: Iterable of field names to preprocess per record.
            filter_non_english: If True, skip records where 'statement' is not English.
        Returns:
            A new list of processed record dicts.
        """
        processed = []
        for rec in records:
            text_to_check = rec.get("statement", "")
            if filter_non_english and not cls.is_english(text_to_check):
                continue
            processed.append(cls.preprocess_record(rec, fields))
        return processed
