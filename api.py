# api.py (Full Code - Refined Logging, Global TextEmbedder, Custom Scores)

import os
import json
import joblib
import torch
from flask import Flask, request, jsonify  # Ensure Flask is imported
import logging  # Import logging
import sys  # For sys.stdout in logger
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# --- Import your custom modules ---
try:
    from preprocessor import Preprocessor
    from code_normalizer import CodeNormalizer
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN  # Assuming this is your filename
    from concat_embedder import ConcatEmbedder
    from submission_predictor import SubmissionPredictor
except ImportError as e:
    # Basic print for critical startup errors before logger is configured
    print(
        f"CRITICAL API IMPORT ERROR: {e}. Ensure all component .py files are accessible and dependencies installed."
    )
    exit()

# --- Load API Configuration ---
try:
    import api_config as cfg
except ImportError:
    print("CRITICAL ERROR: api_config.py not found. Create it with configurations.")
    exit()

try:
    from gemini_client import llm as gemini_llm_instance

    if gemini_llm_instance is None and cfg.ENABLE_LLM_REVIEW:
        print(
            "WARNING: Gemini LLM instance from gemini_client.py is None, but LLM review is enabled. Review will not be generated."
        )
except ImportError as e:
    print(
        f"WARNING: Could not import Gemini LLM client from services/gemini_client.py: {e}. LLM review will be disabled."
    )
    gemini_llm_instance = None
except (
    Exception
) as e_gemini_init:  # Catch other errors from gemini_client.py during its import/init
    print(
        f"ERROR during Gemini LLM client initialization in gemini_client.py: {e_gemini_init}. LLM review will be disabled."
    )
    gemini_llm_instance = None

# --- Global variables to hold loaded components ---
loaded_models: Dict[str, SubmissionPredictor] = {}
loaded_preprocessor: Optional[Preprocessor] = None
loaded_code_normalizer: Optional[CodeNormalizer] = None
loaded_global_text_embedder: Optional[TextEmbedder] = None

app = Flask(__name__)  # Initialize Flask app

load_dotenv()


# --- Configure Flask's app.logger directly ---
def setup_api_logger(flask_app_instance, log_file_path="api_log.txt"):
    """Configures the Flask app's logger for file and console output."""
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):  # Check if log_dir is not empty string
        os.makedirs(log_dir, exist_ok=True)

    # Remove any existing handlers from app.logger to prevent duplicates
    # This is important if app.run(debug=True) was used before or if other libs add handlers
    for handler in list(flask_app_instance.logger.handlers):
        flask_app_instance.logger.removeHandler(handler)

    # Also, stop propagation to the root logger if it has handlers (common cause of duplicates)
    flask_app_instance.logger.propagate = False
    logging.getLogger("werkzeug").propagate = False  # Quiet down Werkzeug if needed

    flask_app_instance.logger.setLevel(logging.INFO)

    # File Handler for app.logger
    try:
        file_handler = logging.FileHandler(log_file_path, mode="a")  # Append mode
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        flask_app_instance.logger.addHandler(file_handler)
    except Exception as e:
        print(
            f"Warning: Could not set up file logger for API at {log_file_path}. Error: {e}"
        )

    # Console Handler for app.logger
    console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use sys.stdout
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [API] - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    flask_app_instance.logger.addHandler(console_handler)

    flask_app_instance.logger.info(
        "Flask app logger configured. Logging to console and file."
    )


def load_inference_components():
    """
    Loads all necessary components for inference.
    Uses app.logger for logging within this function.
    """
    global loaded_models, loaded_preprocessor, loaded_code_normalizer, loaded_global_text_embedder

    app.logger.info("--- Loading inference components (with Global TextEmbedder) ---")
    device = torch.device(cfg.DEVICE_API)

    try:
        loaded_preprocessor = Preprocessor()
        loaded_code_normalizer = CodeNormalizer()
        app.logger.info("Shared Preprocessor and CodeNormalizer initialized.")
    except Exception as e:
        app.logger.error(
            f"Failed to initialize Preprocessor/CodeNormalizer: {e}", exc_info=True
        )
        # Decide if API can run without these. If critical, raise or exit.
        # For now, we'll let it try to continue but predict will likely fail.

    # --- 1. Load the GLOBALLY FITTED TextEmbedder's TfidfVectorizer ---
    app.logger.info(
        f"Loading Global TextEmbedder TfidfVectorizer from: {cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}"
    )
    try:
        if not os.path.exists(cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE):
            msg = f"ERROR: Global TextEmbedder vectorizer file NOT FOUND at '{cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE}'."
            app.logger.error(msg)
            raise FileNotFoundError(msg)  # Raise to stop if critical

        tfidf_vectorizer_fitted = joblib.load(cfg.GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE)
        loaded_global_text_embedder = TextEmbedder(
            vectorizer_model=tfidf_vectorizer_fitted
        )
        loaded_global_text_embedder._fitted = True

        if not (
            hasattr(loaded_global_text_embedder.vectorizer, "vocabulary_")
            and loaded_global_text_embedder.vectorizer.vocabulary_
        ):
            app.logger.warning(
                "Loaded Global TfidfVectorizer does not appear to be fitted (no vocabulary)."
            )
            loaded_global_text_embedder._fitted = False  # Mark as not properly fitted

        app.logger.info(
            f"Global TextEmbedder loaded. Fitted: {loaded_global_text_embedder._fitted}, Vocab size: {len(loaded_global_text_embedder.get_feature_names()) if loaded_global_text_embedder._fitted else 'N/A'}"
        )
    except Exception as e:
        app.logger.error(
            f"Failed to load Global TextEmbedder vectorizer: {e}. Text embedding might not work.",
            exc_info=True,
        )
        if loaded_global_text_embedder is None:
            loaded_global_text_embedder = TextEmbedder()
        app.logger.warning(
            "Created a default, non-fitted Global TextEmbedder as fallback."
        )

    # --- Loop through specified languages to load their specific models ---
    for lang_key in cfg.API_SUPPORTED_LANGUAGES.keys():
        app.logger.info(f"--- Loading language-specific components for: {lang_key} ---")
        try:
            lang_specific_components_dir = os.path.join(
                cfg.SAVED_COMPONENTS_BASE_DIR_API, lang_key
            )
            if not os.path.isdir(lang_specific_components_dir):
                app.logger.warning(
                    f"Directory for language '{lang_key}' not found at '{lang_specific_components_dir}'. Skipping model."
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

            # 2. CodeEmbedderGNN
            app.logger.info(
                f"Loading CodeEmbedderGNN vocabulary from: {code_gnn_vocab_file}"
            )
            if not os.path.exists(code_gnn_vocab_file):
                app.logger.error(
                    f"CodeEmbedderGNN vocabulary file NOT FOUND for {lang_key} at '{code_gnn_vocab_file}'. Skipping model."
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
            app.logger.info(
                f"CodeEmbedderGNN for {lang_key} initialized and vocabulary loaded."
            )

            # 3. ConcatEmbedder
            if (
                loaded_global_text_embedder is None
                or not loaded_global_text_embedder._fitted
            ):
                app.logger.error(
                    f"Global TextEmbedder not ready. Cannot create ConcatEmbedder for {lang_key}."
                )
                continue
            concat_embedder_for_lang = ConcatEmbedder(
                code_embedder=code_embedder_gnn_for_lang,
                text_embedder=loaded_global_text_embedder,
                use_projection=cfg.CONCAT_USE_PROJECTION_API,
                projection_dim_scale_factor=cfg.CONCAT_PROJECTION_SCALE_API,
            ).to(device)
            app.logger.info(
                f"ConcatEmbedder for {lang_key} initialized. Final dim: {concat_embedder_for_lang.final_dim}"
            )

            # 4. SubmissionPredictor
            app.logger.info(f"Loading SubmissionPredictor model from: {model_file}")
            if not os.path.exists(model_file):
                app.logger.error(
                    f"SubmissionPredictor model file NOT FOUND for {lang_key} at '{model_file}'. Skipping model."
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
            app.logger.info(
                f"SubmissionPredictor model for {lang_key} loaded and set to eval mode."
            )
        except Exception as e:
            app.logger.error(
                f"ERROR loading components for {lang_key}: {type(e).__name__} - {e}. This language will not be available.",
                exc_info=True,
            )

    if not loaded_models:
        app.logger.critical(
            "CRITICAL: No models were loaded successfully. API will not make predictions."
        )
    elif loaded_global_text_embedder is None or not loaded_global_text_embedder._fitted:
        app.logger.warning(
            "WARNING: Global TextEmbedder failed. Predictions may be impaired."
        )
    else:
        app.logger.info(
            f"--- API ready. Serving models for languages: {list(loaded_models.keys())} with a global text embedder. ---"
        )


@app.route("/predict", methods=["POST"])
def predict():
    global loaded_models, loaded_preprocessor, loaded_code_normalizer
    # Global text embedder is used internally by the ConcatEmbedder within the selected_model

    # --- Component Availability Checks ---
    critical_component_missing = False
    if not loaded_models:
        app.logger.error("/predict: No models loaded.")
        critical_component_missing = True
    if not loaded_preprocessor or not loaded_code_normalizer:
        app.logger.error("/predict: Preprocessors not loaded.")
        critical_component_missing = True
    if not loaded_global_text_embedder or not loaded_global_text_embedder._fitted:
        app.logger.error("/predict: Global TextEmbedder not ready.")
        critical_component_missing = True
    if cfg.ENABLE_LLM_REVIEW and gemini_llm_instance is None:
        app.logger.warning(
            "/predict: LLM review enabled in config, but LLM instance is not available."
        )
        # Continue without LLM review for this request, or return error if LLM is critical
    if critical_component_missing:
        return jsonify({"error": "Core API components not ready."}), 503

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
            field is not None for field in [statement, code_submission, language]
        ):
            return (
                jsonify(
                    {"error": "Missing fields: statement, code_submission, language."}
                ),
                400,
            )

        requested_lang_normalized = language.strip().lower()
        selected_model = loaded_models.get(requested_lang_normalized)

        if not selected_model:
            return (
                jsonify(
                    {
                        "error": f"Unsupported language '{language}'. Available: {list(loaded_models.keys())}."
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

        # 2. Model Prediction
        with torch.no_grad():
            verdict_logits = selected_model(
                [normalized_code], [processed_statement], requested_lang_normalized
            )

        # 3. Process output & Calculate Scores
        probabilities_tensor = torch.softmax(verdict_logits, dim=1).squeeze()
        predicted_id = torch.argmax(probabilities_tensor).item()
        predicted_verdict_str = cfg.ID_TO_VERDICT_MAP.get(
            predicted_id, f"UnmappedID_{predicted_id}"
        )

        verdict_probs_map: Dict[str, float] = {
            cfg.ID_TO_VERDICT_MAP.get(i, f"Cls_{i}"): prob.item()
            for i, prob in enumerate(probabilities_tensor)
        }

        # --- LLM Review Generation ---
        llm_review = "LLM review disabled or LLM not available."
        if cfg.ENABLE_LLM_REVIEW and gemini_llm_instance:
            try:
                app.logger.info(
                    f"Generating LLM review for lang: {language}, verdict: {predicted_verdict_str}"
                )
                # Format probabilities for the prompt
                verdict_probabilities_str_for_prompt = "\n".join(
                    [
                        f"- {v_name}: {prob:.2%}"
                        for v_name, prob in verdict_probs_map.items()
                    ]
                )

                prompt = cfg.LLM_REVIEW_PROMPT_TEMPLATE.format(
                    language=language,
                    predicted_verdict_string=predicted_verdict_str,
                    verdict_probabilities_str=verdict_probabilities_str_for_prompt,
                )

                # Langchain specific invocation
                from langchain_core.messages import (
                    HumanMessage,
                )  # Assuming HumanMessage is appropriate

                llm_response = gemini_llm_instance.invoke(
                    [HumanMessage(content=prompt)]
                )
                llm_review = (
                    llm_response.content
                    if hasattr(llm_response, "content")
                    else str(llm_response)
                )
                app.logger.info(
                    f"LLM Review Generated: {llm_review[:100]}..."
                )  # Log first 100 chars
            except Exception as e_llm:
                app.logger.error(
                    f"Error during LLM review generation: {e_llm}", exc_info=True
                )
                llm_review = "Error generating LLM review."

        response = {
            "requested_language": language,
            "predicted_verdict_id": predicted_id,
            "predicted_verdict_string": predicted_verdict_str,
            "verdict_probabilities": verdict_probs_map,
            "llm_review": llm_review,  # Add LLM review to the response
        }
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(
            f"Error during /predict for lang '{language if 'language' in locals() else 'unknown'}': {e}",
            exc_info=True,
        )
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # --- Setup Flask App Logger ---
    # Define where API logs will go (can be different from training logs)
    API_LOG_FILE = os.path.join(
        cfg.LOG_DIR if hasattr(cfg, "LOG_DIR") else "training_logs",
        "api_runtime_log.txt",
    )
    if not hasattr(cfg, "LOG_DIR") and not os.path.exists("training_logs"):
        os.makedirs("training_logs", exist_ok=True)  # Ensure default log dir exists

    setup_api_logger(app, API_LOG_FILE)  # Configure app.logger

    try:
        load_inference_components()  # Load models when the app starts

        ready_to_run = True
        if not loaded_models:
            app.logger.critical("API cannot start: no models were successfully loaded.")
            ready_to_run = False
        if not loaded_global_text_embedder or not loaded_global_text_embedder._fitted:
            app.logger.warning(
                "API WARNING: Global TextEmbedder failed to load or is not fitted. Predictions will likely fail."
            )
            # Depending on strictness, you might set ready_to_run = False here too

        if ready_to_run:
            app.logger.info("Starting Flask development server for API...")
            # For development: debug=True, use_reloader=True (but can cause double logging/init)
            # For this controlled setup: debug=False, use_reloader=False to rely on our logger

            app.run(
                debug=False,
                host="0.0.0.0",
                port=os.environ.get("PORT"),
                use_reloader=False,
            )
        else:
            app.logger.info(
                "Flask app not started due to critical component loading failures."
            )

    except Exception as startup_error:
        # Use app.logger if available, else print
        if hasattr(app, "logger") and app.logger.handlers:  # Check if handlers are set
            app.logger.critical(
                f"FATAL ERROR during API startup: {startup_error}", exc_info=True
            )
        else:
            print(
                f"FATAL ERROR during API startup (app.logger not fully configured): {startup_error}"
            )
            import traceback

            traceback.print_exc()
