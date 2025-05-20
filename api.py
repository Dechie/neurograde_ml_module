import os
import json
import joblib
import torch
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, List

# --- Import your custom modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN  # Corrected import
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
    print("CRITICAL ERROR: api_config.py not found. Create it with configurations.")
    exit()

loaded_models: Dict[str, SubmissionPredictor] = {}
loaded_preprocessor: Optional[Preprocessor] = None
loaded_code_normalizer: Optional[CodeNormalizer] = None
# --- Global TextEmbedder instance ---
loaded_global_text_embedder: Optional[TextEmbedder] = None

app = Flask(__name__)


def load_inference_components():
    global loaded_models, loaded_preprocessor, loaded_code_normalizer, loaded_global_text_embedder

    print("--- Loading inference components (with Global TextEmbedder) ---")
    device = torch.device(cfg.DEVICE_API)

    loaded_preprocessor = Preprocessor()
    loaded_code_normalizer = CodeNormalizer()
    print("Shared Preprocessor and CodeNormalizer initialized.")

    # --- 1. Load the GLOBALLY FITTED TextEmbedder's TfidfVectorizer ---
    print(
        f"Loading Global TextEmbedder TfidfVectorizer from: {cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}"
    )
    try:
        if not os.path.exists(cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            print(
                f"ERROR: Global TextEmbedder vectorizer file NOT FOUND at '{cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}'. API cannot effectively start for text features."
            )
            raise FileNotFoundError  # Or handle more gracefully by disabling text features

        tfidf_vectorizer_fitted = joblib.load(cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE)
        loaded_global_text_embedder = TextEmbedder(
            vectorizer_model=tfidf_vectorizer_fitted
        )
        loaded_global_text_embedder._fitted = (
            True  # Assume joblib restores a fitted model
        )

        if not (
            hasattr(loaded_global_text_embedder.vectorizer, "vocabulary_")
            and loaded_global_text_embedder.vectorizer.vocabulary_
        ):
            print(
                f"WARNING: Loaded Global TfidfVectorizer does not appear to be fitted (no vocabulary)."
            )
            loaded_global_text_embedder._fitted = False

        print(
            f"Global TextEmbedder loaded. Fitted: {loaded_global_text_embedder._fitted}, Vocab size: {len(loaded_global_text_embedder.get_feature_names()) if loaded_global_text_embedder._fitted else 'N/A'}"
        )
    except Exception as e:
        print(
            f"ERROR: Failed to load Global TextEmbedder vectorizer: {e}. Text embedding will not work."
        )
        # loaded_global_text_embedder will remain None or in an unfitted state
        # Subsequent ConcatEmbedder initializations might warn or fail if they strictly need a fitted TextEmbedder dimension.
        # For robustness, we could create a dummy unfitted TextEmbedder here.
        if loaded_global_text_embedder is None:
            loaded_global_text_embedder = TextEmbedder()  # Create a non-fitted instance
            print("Created a default, non-fitted Global TextEmbedder as fallback.")

    for lang_key in cfg.API_SUPPORTED_LANGUAGES.keys():
        print(f"\n--- Loading language-specific components for: {lang_key} ---")
        try:
            lang_specific_components_dir = os.path.join(
                cfg.SAVED_COMPONENTS_BASE_DIR_API, lang_key
            )
            if not os.path.isdir(lang_specific_components_dir):
                print(
                    f"  WARNING: Directory for language '{lang_key}' not found at '{lang_specific_components_dir}'. Skipping this language model."
                )
                continue

            model_file = os.path.join(
                lang_specific_components_dir,
                cfg.MODEL_FILENAME_TPL.format(lang_key=lang_key),
            )
            code_gnn_vocab_file = os.path.join(
                lang_specific_components_dir,
                cfg.CODE_GNN_VOCAB_FILENAME_TPL.format(lang_key=lang_key),
            )

            # TextEmbedder is now global, no need to load it per language.
            # We will pass `loaded_global_text_embedder` to each ConcatEmbedder.

            # 2. Load CodeEmbedderGNN's vocabulary and initialize for this language
            print(f"  Loading CodeEmbedderGNN vocabulary from: {code_gnn_vocab_file}")
            if not os.path.exists(code_gnn_vocab_file):
                print(
                    f"  ERROR: CodeEmbedderGNN vocabulary file NOT FOUND for {lang_key}. Skipping this language model."
                )
                continue
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

            # 3. Initialize ConcatEmbedder for this language, using the GLOBAL TextEmbedder
            if (
                loaded_global_text_embedder is None
                or not loaded_global_text_embedder._fitted
            ):
                print(
                    f"  ERROR: Global TextEmbedder is not available or not fitted. Cannot create ConcatEmbedder for {lang_key}."
                )
                continue  # Skip this language model if global text embedder is problematic

            concat_embedder_for_lang = ConcatEmbedder(
                code_embedder=code_embedder_gnn_for_lang,
                text_embedder=loaded_global_text_embedder,  # Use the global instance
                use_projection=cfg.CONCAT_USE_PROJECTION_API,
                projection_dim_scale_factor=cfg.CONCAT_PROJECTION_SCALE_API,
            ).to(device)
            print(
                f"  ConcatEmbedder for {lang_key} initialized. Final dim: {concat_embedder_for_lang.final_dim}"
            )

            # 4. Initialize SubmissionPredictor and load trained weights for this language
            print(f"  Loading SubmissionPredictor model from: {model_file}")
            if not os.path.exists(model_file):
                print(
                    f"  ERROR: SubmissionPredictor model file NOT FOUND for {lang_key}. Skipping this language model."
                )
                continue
            model_instance_for_lang = SubmissionPredictor(
                concat_embedder=concat_embedder_for_lang,
                num_verdict_classes=cfg.NUM_VERDICT_CLASSES_API,
                mlp_hidden_dims=cfg.PREDICTOR_MLP_HIDDEN_DIMS_API,
            ).to(device)
            model_instance_for_lang.load_state_dict(
                torch.load(model_file, map_location=device)
            )
            model_instance_for_lang.eval()

            loaded_models[lang_key] = model_instance_for_lang
            print(
                f"  SubmissionPredictor model for {lang_key} loaded and set to eval mode."
            )

        except Exception as e:
            print(
                f"  ERROR loading components for {lang_key}: {type(e).__name__} - {e}. This language will not be available."
            )
            import traceback

            traceback.print_exc()

    if not loaded_models:
        print(
            "CRITICAL: No models were loaded successfully. The API will not be able to make predictions."
        )
    elif loaded_global_text_embedder is None or not loaded_global_text_embedder._fitted:
        print(
            "WARNING: Global TextEmbedder failed to load or is not fitted. Predictions might be impaired or fail."
        )
    else:
        print(
            f"\n--- API ready. Serving models for languages: {list(loaded_models.keys())} with a global text embedder. ---"
        )


# The /predict endpoint and if __name__ == "__main__": block remain unchanged.
# Only `load_inference_components` needed adjustment for how TextEmbedder is handled.
# ... (rest of api.py: @app.route("/predict") and if __name__ == "__main__": from previous version)


@app.route("/predict", methods=["POST"])
def predict():
    global loaded_models, loaded_preprocessor, loaded_code_normalizer, loaded_global_text_embedder

    if not loaded_models:
        return jsonify({"error": "No models loaded. API is not ready."}), 500
    if not loaded_preprocessor or not loaded_code_normalizer:
        return jsonify({"error": "Preprocessors not loaded. API is not ready."}), 500
    if not loaded_global_text_embedder or not loaded_global_text_embedder._fitted:
        return (
            jsonify(
                {
                    "error": "Global TextEmbedder not loaded or not fitted. API cannot make predictions."
                }
            ),
            500,
        )

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        statement = data.get("statement")
        input_spec = data.get("input_spec", "")
        output_spec = data.get("output_spec", "")
        code_submission = data.get("code_submission")
        language = data.get("language")

        if not all(
            [statement is not None, code_submission is not None, language is not None]
        ):  # Check for presence
            return (
                jsonify(
                    {
                        "error": "Missing required fields: statement, code_submission, language."
                    }
                ),
                400,
            )

        requested_lang_normalized = language.strip().lower()
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

        full_statement_raw = (
            str(statement).strip()
            + "\nInput: "
            + str(input_spec).strip()
            + "\nOutput: "
            + str(output_spec).strip()
        )
        # Preprocessor is global
        processed_statement = loaded_preprocessor.preprocess_text(full_statement_raw)
        # CodeNormalizer is global
        normalized_code = loaded_code_normalizer.normalize_code(
            str(code_submission), requested_lang_normalized
        )

        with torch.no_grad():
            # selected_model already contains its ConcatEmbedder, which contains
            # its CodeEmbedderGNN and the shared loaded_global_text_embedder
            verdict_logits = selected_model(
                code_list=[normalized_code],
                text_list=[processed_statement],
                lang=requested_lang_normalized,
            )

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
            print(
                "API cannot start as no models were successfully loaded. Check logs and file paths in api_config.py."
            )
        elif not loaded_global_text_embedder or not loaded_global_text_embedder._fitted:
            print(
                "API may have issues as Global TextEmbedder failed to load or is not fitted."
            )
            app.run(
                debug=True, host="0.0.0.0", port=5000
            )  # Still run if some models loaded but text embedder is an issue
        else:
            app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as startup_error:
        print(f"FATAL ERROR during API startup: {startup_error}")
        import traceback

        traceback.print_exc()
