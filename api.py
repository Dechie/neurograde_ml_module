# api.py (Updated for Multi-Language Support)

import os
import json
import joblib
import torch
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, List

try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN
    from concat_embedder import ConcatEmbedder
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    print(
        f"CRITICAL API IMPORT ERROR: {e}. Ensure all component .py files are accessible."
    )
    exit()

# --- Load API Configuration ---
try:
    import api_config as cfg
except ImportError:
    print(
        "CRITICAL ERROR: api_config.py not found. Please create it with necessary configurations."
    )
    exit()

# --- Global variables to hold loaded components ---
# Models and associated components will be keyed by language (lowercase)
loaded_models: Dict[str, SubmissionPredictor] = (
    {}
)  # Stores a SubmissionPredictor for each language
# Preprocessor and CodeNormalizer can be shared if they are language-agnostic enough
# or if their methods correctly handle different languages.
loaded_preprocessor: Optional[Preprocessor] = None
loaded_code_normalizer: Optional[CodeNormalizer] = None

app = Flask(__name__)


def load_inference_components():
    """
    Loads all necessary components FOR EACH SUPPORTED LANGUAGE for inference.
    This function is called once when the Flask app starts.
    """
    global loaded_models, loaded_preprocessor, loaded_code_normalizer

    print("--- Loading inference components for multiple languages ---")
    device = torch.device(cfg.DEVICE_API)

    # Initialize shared Preprocessor and CodeNormalizer (if they are truly shared)
    loaded_preprocessor = Preprocessor()
    loaded_code_normalizer = CodeNormalizer()
    print("Shared Preprocessor and CodeNormalizer initialized.")

    for lang_key, lang_value_for_filename in cfg.API_SUPPORTED_LANGUAGES.items():
        print(f"\n--- Loading components for language: {lang_key} ---")
        try:
            # Construct file paths for the current language
            model_file = cfg.MODEL_FILE_TPL.format(
                lang_key=lang_key, lang_value=lang_value_for_filename
            )
            text_embedder_vectorizer_file = (
                cfg.TEXT_EMBEDDER_VECTORIZER_FILE_TPL.format(
                    lang_key=lang_key, lang_value=lang_value_for_filename
                )
            )
            code_gnn_vocab_file = cfg.CODE_GNN_VOCAB_FILE_TPL.format(
                lang_key=lang_key, lang_value=lang_value_for_filename
            )

            # 1. Load TextEmbedder's TfidfVectorizer for this language
            print(
                f"  Loading TextEmbedder TfidfVectorizer from: {text_embedder_vectorizer_file}"
            )
            tfidf_vectorizer_fitted = joblib.load(text_embedder_vectorizer_file)
            text_embedder_for_lang = TextEmbedder(
                vectorizer_model=tfidf_vectorizer_fitted
            )
            text_embedder_for_lang._fitted = (
                True  # Assume joblib restores a fitted model
            )
            if (
                not hasattr(text_embedder_for_lang.vectorizer, "vocabulary_")
                or not text_embedder_for_lang.vectorizer.vocabulary_
            ):
                print(
                    f"  WARNING: Loaded TfidfVectorizer for {lang_key} does not appear to be fitted."
                )
                text_embedder_for_lang._fitted = False  # Correct if no vocab

            print(
                f"  TextEmbedder for {lang_key} loaded. Fitted: {text_embedder_for_lang._fitted}, Vocab size: {len(text_embedder_for_lang.get_feature_names()) if text_embedder_for_lang._fitted else 'N/A'}"
            )

            # 2. Load CodeEmbedderGNN's vocabulary and initialize for this language
            print(f"  Loading CodeEmbedderGNN vocabulary from: {code_gnn_vocab_file}")
            with open(code_gnn_vocab_file, "r") as f:
                code_gnn_vocab_data = json.load(f)

            code_embedder_gnn_for_lang = CodeEmbedderGNN(
                node_vocab_size=code_gnn_vocab_data.get(
                    "node_vocab_size", cfg.CODE_GNN_NODE_VOCAB_SIZE_API
                ),
                node_embedding_dim=cfg.CODE_GNN_NODE_EMB_DIM_API,
                hidden_gnn_dim=cfg.CODE_GNN_HIDDEN_DIM_API,
                out_graph_embedding_dim=cfg.CODE_GNN_OUT_DIM_API,
                num_gnn_layers=cfg.CODE_GNN_LAYERS_API,
            ).to(device)
            code_embedder_gnn_for_lang.node_type_to_id = code_gnn_vocab_data[
                "node_type_to_id"
            ]
            code_embedder_gnn_for_lang.next_node_type_id = code_gnn_vocab_data[
                "next_node_type_id"
            ]
            print(
                f"  CodeEmbedderGNN for {lang_key} initialized and vocabulary loaded."
            )

            # 3. Initialize ConcatEmbedder for this language
            concat_embedder_for_lang = ConcatEmbedder(
                code_embedder=code_embedder_gnn_for_lang,
                text_embedder=text_embedder_for_lang,
                use_projection=cfg.CONCAT_USE_PROJECTION_API,
                projection_dim_scale_factor=cfg.CONCAT_PROJECTION_SCALE_API,
            ).to(device)
            print(
                f"  ConcatEmbedder for {lang_key} initialized. Final dim: {concat_embedder_for_lang.final_dim}"
            )

            # 4. Initialize SubmissionPredictor and load trained weights for this language
            print(f"  Loading SubmissionPredictor model from: {model_file}")
            model_instance_for_lang = SubmissionPredictor(
                concat_embedder=concat_embedder_for_lang,
                num_verdict_classes=cfg.NUM_VERDICT_CLASSES_API,
                mlp_hidden_dims=cfg.PREDICTOR_MLP_HIDDEN_DIMS_API,
            ).to(device)
            model_instance_for_lang.load_state_dict(
                torch.load(model_file, map_location=device)
            )
            model_instance_for_lang.eval()

            loaded_models[lang_key] = model_instance_for_lang  # Store in the dictionary
            print(
                f"  SubmissionPredictor model for {lang_key} loaded and set to eval mode."
            )

        except FileNotFoundError as e:
            print(
                f"  ERROR loading components for {lang_key}: File not found - {e}. This language will not be available."
            )
        except Exception as e:
            print(
                f"  ERROR loading components for {lang_key}: {type(e).__name__} - {e}. This language will not be available."
            )
            import traceback

            traceback.print_exc()  # Print stack trace for detailed debugging

    if not loaded_models:
        print(
            "CRITICAL: No models were loaded successfully. The API will not be able to make predictions."
        )
    else:
        print(
            f"\n--- API ready. Serving models for languages: {list(loaded_models.keys())} ---"
        )


@app.route("/predict", methods=["POST"])
def predict():
    global loaded_models, loaded_preprocessor, loaded_code_normalizer  # loaded_models is now a dict

    if not loaded_models:  # Check if the dictionary is empty
        return jsonify({"error": "No models loaded. API is not ready."}), 500
    if not loaded_preprocessor or not loaded_code_normalizer:
        return jsonify({"error": "Preprocessors not loaded. API is not ready."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        statement = data.get("statement")
        input_spec = data.get("input_spec", "")  # Default to empty string
        output_spec = data.get("output_spec", "")  # Default to empty string
        code_submission = data.get("code_submission")
        language = data.get("language")

        if not all([statement, code_submission, language]):
            return (
                jsonify(
                    {
                        "error": "Missing required fields: statement, code_submission, language."
                    }
                ),
                400,
            )

        requested_lang_normalized = language.strip().lower()

        # --- Select the correct model for the requested language ---
        selected_model = loaded_models.get(requested_lang_normalized)
        if not selected_model:
            return (
                jsonify(
                    {
                        "error": f"Unsupported language '{language}'. This API instance serves: {list(loaded_models.keys())}.",
                        "requested_language": language,
                        "available_languages": list(loaded_models.keys()),
                    }
                ),
                400,
            )

        # 1. Preprocessing
        full_statement_raw = (
            str(statement).strip()
            + "\nInput: "
            + str(input_spec).strip()
            + "\nOutput: "
            + str(output_spec).strip()
        )
        processed_statement = loaded_preprocessor.preprocess_text(full_statement_raw)
        normalized_code = loaded_code_normalizer.normalize_code(
            str(code_submission), requested_lang_normalized
        )

        # 2. Model Prediction using the selected model
        with torch.no_grad():
            verdict_logits = selected_model(  # Use the language-specific model
                code_list=[normalized_code],
                text_list=[processed_statement],
                lang=requested_lang_normalized,  # Pass the lang to the model's ConcatEmbedder
            )

        # 3. Process output
        probabilities = torch.softmax(verdict_logits, dim=1).squeeze()
        predicted_id = torch.argmax(probabilities).item()
        predicted_verdict_str = cfg.ID_TO_VERDICT_MAP.get(
            predicted_id, "Error: Unmapped Verdict ID"
        )
        class_probabilities = {
            cfg.ID_TO_VERDICT_MAP.get(i, f"Class_{i}"): prob.item()
            for i, prob in enumerate(probabilities)
        }

        response = {
            "requested_language": language,
            "predicted_verdict_id": predicted_id,
            "predicted_verdict_string": predicted_verdict_str,
            "verdict_probabilities": class_probabilities,
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    try:
        load_inference_components()
        if not loaded_models:
            print("API cannot start as no models were successfully loaded. Check logs.")
        else:
            app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as startup_error:
        print(f"FATAL ERROR during API startup: {startup_error}")
        import traceback

        traceback.print_exc()
