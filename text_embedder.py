# text_embedder.py

import sys
from typing import List, Optional, Union, Dict, Any  # Added Dict, Any for consistency

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import NotFittedError
except ImportError:
    sys.exit(
        "ERROR: scikit-learn is not installed. TextEmbedder requires TfidfVectorizer. "
        "Please install it by running: pip install scikit-learn"
    )

import numpy as np

# __all__ = ["TextEmbedder"] # You can keep this if it's part of a larger package structure


class TextEmbedder:
    """
    A class for converting preprocessed text into vector representations using TF-IDF.
    """

    def __init__(
        self,
        vectorizer_model: Optional[TfidfVectorizer] = None,
        max_features: Optional[int] = 5000,  # Common param to expose
        ngram_range: tuple = (1, 1),  # Common param to expose
        min_df: int = 1,  # Common param to expose
        max_df: float = 1.0,
    ):  # Common param to expose
        """
        Initializes the TextEmbedder.

        Args:
            vectorizer_model: An optional, pre-configured/pre-fitted TfidfVectorizer.
                              If provided, other TF-IDF parameters are ignored.
            max_features: If vectorizer_model is None, the maximum number of
                          features (vocabulary size) for the TF-IDF vectorizer.
            ngram_range: If vectorizer_model is None, the N-gram range for TF-IDF.
            min_df: If vectorizer_model is None, min document frequency for TF-IDF.
            max_df: If vectorizer_model is None, max document frequency for TF-IDF.
        """
        if vectorizer_model:
            if not isinstance(vectorizer_model, TfidfVectorizer):
                raise TypeError(
                    "Provided 'vectorizer_model' must be an instance of TfidfVectorizer."
                )
            self.vectorizer: TfidfVectorizer = vectorizer_model
            # Check if the provided model is already fitted
            try:
                # Accessing vocabulary_ is a common way to check if fitted
                if (
                    hasattr(self.vectorizer, "vocabulary_")
                    and self.vectorizer.vocabulary_
                ):
                    self._fitted = True
                else:
                    self._fitted = False
            except NotFittedError:
                self._fitted = False
        else:
            self.vectorizer: TfidfVectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )
            self._fitted = False

        # For compatibility with original class structure from the problem description
        # We can get the feature dimension after fitting.
        # self.out_dim = None # Will be set after fitting

    def fit(self, texts: List[str]) -> None:
        """
        Fits the TF-IDF vectorizer on a list of preprocessed text strings.

        Args:
            texts: A list of strings to learn the vocabulary from.
        """
        if not texts:
            print(
                "Warning (TextEmbedder.fit): Received empty list of texts. Vectorizer will not be fitted."
            )
            return
        try:
            self.vectorizer.fit(texts)
            self._fitted = True
            # self.out_dim = len(self.get_feature_names()) # Set output dimension
        except Exception as e:
            print(f"Error (TextEmbedder.fit): Failed to fit vectorizer. Reason: {e}")
            self._fitted = False
            # self.out_dim = None
            raise  # Re-raise the exception so the caller knows fitting failed

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforms a list of preprocessed texts into TF-IDF vectors.

        Args:
            texts: A list of preprocessed text strings.

        Returns:
            A 2D NumPy array of shape (len(texts), vocabulary_size).

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() or fit_transform() first."
            )
        if not texts:  # Handle empty input list for transform
            # Output dimension depends on whether it's fitted.
            # If fitted, vocab size is known. If not, this line shouldn't be reached.
            vocab_size = (
                len(self.vectorizer.vocabulary_)
                if hasattr(self.vectorizer, "vocabulary_")
                else 0
            )
            return np.empty((0, vocab_size), dtype=float)
        try:
            return self.vectorizer.transform(texts).toarray()
        except Exception as e:
            print(
                f"Error (TextEmbedder.transform): Failed to transform texts. Reason: {e}"
            )
            raise

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fits the vectorizer on the texts and then transforms them.

        Args:
            texts: A list of preprocessed text strings.

        Returns:
            A 2D NumPy array of TF-IDF vectors.
        """
        if not texts:
            print(
                "Warning (TextEmbedder.fit_transform): Received empty list of texts. Vectorizer will not be fitted and empty array returned."
            )
            # self.out_dim = 0
            return np.empty(
                (0, 0), dtype=float
            )  # Or choose a different strategy for unfitted + no text
        try:
            result = self.vectorizer.fit_transform(texts).toarray()
            self._fitted = True
            # self.out_dim = result.shape[1] if result.ndim == 2 and result.shape[0] > 0 else 0
            return result
        except Exception as e:
            print(
                f"Error (TextEmbedder.fit_transform): Failed to fit_transform texts. Reason: {e}"
            )
            self._fitted = False
            # self.out_dim = None
            raise

    def get_feature_names(self) -> List[str]:
        """
        Returns the feature names (vocabulary) of the TF-IDF model.

        Returns:
            A list of strings, where each string is a feature name.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() or fit_transform() first."
            )
        try:
            return self.vectorizer.get_feature_names_out().tolist()
        except NotFittedError:  # Should be caught by self._fitted, but as a safeguard
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() or fit_transform() first."
            )
        except Exception as e:
            print(
                f"Error (TextEmbedder.get_feature_names): Could not get feature names. Reason: {e}"
            )
            raise

    # The embed_record and embed_records methods are slightly different from the prompt's class structure
    # The prompt expected them as direct methods using the internal vectorizer.
    # For `embed_record(record, field)` and `embed_records(records, field)`
    # These methods in your original TextEmbedder are fine and directly use self.transform.

    def embed_record(
        self, record: Dict[str, Any], field: str = "statement"
    ) -> Optional[np.ndarray]:
        """
        Embeds a single preprocessed text field from a dictionary record.

        Args:
            record: A dictionary containing the text field to embed.
            field: The key of the text field in the record.

        Returns:
            A 1D NumPy array representing the TF-IDF vector, or None if the
            field is missing or an error occurs.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() first for embed_record."
            )

        text_to_embed = record.get(field)
        if text_to_embed is None or not isinstance(text_to_embed, str):
            # print(f"Warning (embed_record): Field '{field}' not found or not a string in record. Returning None.")
            # Return zero vector of appropriate dimension if field is missing or NaN
            # This assumes you want a consistent output shape for batches later.
            vocab_size = (
                len(self.vectorizer.vocabulary_)
                if hasattr(self.vectorizer, "vocabulary_")
                else 0
            )
            return (
                np.zeros(vocab_size) if self._fitted else None
            )  # Return None if not fitted to avoid guessing dim
            # Or return None as you originally had:
            # return None

        # Transform expects a list of documents
        try:
            # Ensure text_to_embed is not empty, TFIDF might behave differently
            if not text_to_embed.strip():  # If it's an empty or whitespace-only string
                vocab_size = len(self.vectorizer.vocabulary_)
                return np.zeros(vocab_size)
            return self.transform([text_to_embed])[0]
        except Exception as e:
            print(
                f"Error (TextEmbedder.embed_record): Failed to embed record for field '{field}'. Reason: {e}"
            )
            return None

    def embed_records(
        self, records: List[Dict[str, Any]], field: str = "statement"
    ) -> Optional[np.ndarray]:
        """
        Embeds a specific text field from all records in a list of dictionaries.
        Records missing the specified field or having non-string content for it
        will result in a row of zeros in the output array.

        Args:
            records: A list of dictionaries.
            field: The key of the text field to embed from each record.

        Returns:
            A 2D NumPy array where each row is the TF-IDF vector for a record.
            Returns None if a fatal error occurs or the vectorizer is not fitted.

        Raises:
            RuntimeError: If the vectorizer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Vectorizer has not been fitted. Call fit() first for embed_records."
            )

        if not records:
            vocab_size = (
                len(self.vectorizer.vocabulary_)
                if hasattr(self.vectorizer, "vocabulary_")
                else 0
            )
            return np.empty((0, vocab_size), dtype=float)

        texts_to_embed: List[str] = []
        for record in records:
            text_val = record.get(field)
            if isinstance(text_val, str):
                texts_to_embed.append(
                    text_val if text_val.strip() else ""
                )  # Use empty string for TF-IDF to get zero vector
            else:
                texts_to_embed.append(
                    ""
                )  # Treat missing or non-string as empty for consistent zero vector

        try:
            return self.transform(texts_to_embed)
        except Exception as e:
            print(
                f"Error (TextEmbedder.embed_records): Failed to embed records for field '{field}'. Reason: {e}"
            )
            return None  # Or re-raise


# --- Example Usage ---
if __name__ == "__main__":
    print("--- TextEmbedder Demo ---")

    sample_corpus = [
        "this is the first document",
        "this document is the second document",
        "and this is the third one",
        "is this the first document again",
    ]

    # Test 1: Default initialization and full fit_transform
    print("\nTest 1: Default Initialization & fit_transform")
    try:
        embedder1 = TextEmbedder(max_features=10)  # Limit features for easier display
        embeddings1 = embedder1.fit_transform(sample_corpus)
        print(f"  Embeddings shape: {embeddings1.shape}")
        print(f"  Vocabulary (first 5): {embedder1.get_feature_names()[:5]}")
        print(f"  Is fitted: {embedder1._fitted}")
    except Exception as e:
        print(f"  Error in Test 1: {e}")

    # Test 2: Separate fit and transform
    print("\nTest 2: Separate fit and transform")
    try:
        embedder2 = TextEmbedder(max_features=10)
        embedder2.fit(sample_corpus)
        print(f"  Is fitted after fit(): {embedder2._fitted}")
        print(f"  Vocabulary (first 5): {embedder2.get_feature_names()[:5]}")
        single_doc_embedding = embedder2.transform(
            ["this is a new document to transform"]
        )
        print(f"  Single doc embedding shape: {single_doc_embedding.shape}")
    except Exception as e:
        print(f"  Error in Test 2: {e}")

    # Test 3: Using embed_record and embed_records
    print("\nTest 3: embed_record and embed_records")
    if "embedder1" in locals() and embedder1._fitted:
        records_data = [
            {"id": 1, "statement": "first document text"},
            {"id": 2, "statement": "second document text here"},
            {"id": 3, "another_field": "not a statement"},  # Missing 'statement'
            {"id": 4, "statement": ""},  # Empty statement
            {"id": 5, "statement": "document"},
        ]
        try:
            print(f"  Embedding single record (id 1):")
            record1_emb = embedder1.embed_record(records_data[0], field="statement")
            if record1_emb is not None:
                print(f"    Shape: {record1_emb.shape}")

            print(f"  Embedding single record with missing field (id 3):")
            record3_emb = embedder1.embed_record(records_data[2], field="statement")
            if record3_emb is not None:  # Should be a zero vector
                print(
                    f"    Shape: {record3_emb.shape}, Is all zero: {np.all(record3_emb == 0)}"
                )

            print(f"  Embedding all records:")
            all_records_emb = embedder1.embed_records(records_data, field="statement")
            if all_records_emb is not None:
                print(
                    f"    Shape: {all_records_emb.shape}"
                )  # Should be (5, num_features)
                # Check if the row for the missing statement (index 2) is all zeros
                if all_records_emb.shape[0] == len(records_data):
                    print(
                        f"    Row for missing field (idx 2) is all zero: {np.all(all_records_emb[2] == 0)}"
                    )
                    print(
                        f"    Row for empty field (idx 3) is all zero: {np.all(all_records_emb[3] == 0)}"
                    )

        except Exception as e:
            print(f"  Error in Test 3: {e}")
    else:
        print("  Skipping Test 3 because embedder1 was not successfully fitted.")

    # Test 4: Passing a pre-configured TfidfVectorizer
    print("\nTest 4: Passing pre-configured TfidfVectorizer")
    try:
        custom_tfidf = TfidfVectorizer(max_features=5, stop_words="english")
        embedder4 = TextEmbedder(vectorizer_model=custom_tfidf)
        # Fit it using the embedder's method
        embeddings4 = embedder4.fit_transform(sample_corpus)
        print(f"  Embeddings shape: {embeddings4.shape}")
        print(f"  Vocabulary (custom): {embedder4.get_feature_names()}")
    except Exception as e:
        print(f"  Error in Test 4: {e}")

    # Test 5: Error handling for calling transform before fit
    print("\nTest 5: Transform before fit")
    try:
        embedder5 = TextEmbedder()
        embedder5.transform(["this should fail"])
    except RuntimeError as e:
        print(f"  Caught expected error: {e}")
    except Exception as e:
        print(f"  Caught unexpected error in Test 5: {e}")

    print("\n--- TextEmbedder Demo Finished ---")
