import csv
import re
import string
import nltk
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Regex matching any Japanese character (Hiragana, Katakana, Kanji)
jp_regex = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", re.UNICODE)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\$\$\$|\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(
        stemmer.stem(tok) for tok in tokens if tok.isalpha() and tok not in stop_words
    )


def is_english(text):
    """
    Returns True if 'text' is likely English via:
    1) langdetect → 'en'
    2) no Japanese characters via regex
    """
    # Quick regex check to skip obvious Japanese
    if jp_regex.search(text):
        return False

    # Fallback to langdetect
    try:
        return detect(text) == "en"
    except LangDetectException:
        # If detection fails (too short/etc.), assume non-English
        return False


def preprocess_csv(input_csv, output_csv):
    processed = []
    with open(input_csv, newline="", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        for row in reader:
            # Only keep rows whose *original* statement is English
            if not is_english(row.get("statement", "")):
                continue

            # Preprocess all three text fields
            for field in ("statement", "input_spec", "output_spec"):
                row[field] = preprocess_text(row.get(field, ""))

            processed.append(
                {
                    "problem_id": row.get("problem_id", ""),
                    "statement": row["statement"],
                    "input_spec": row["input_spec"],
                    "output_spec": row["output_spec"],
                }
            )

    # Write filtered & preprocessed data
    with open(output_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(
            wf,
            fieldnames=["problem_id", "statement", "input_spec", "output_spec"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(processed)


if __name__ == "__main__":
    preprocess_csv(
        input_csv="problems_statements.csv", output_csv="problems_preprocessed.csv"
    )
    print("✅ Saved filtered & preprocessed data to 'problems_preprocessed.csv'")
