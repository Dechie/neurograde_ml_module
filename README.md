# Automated Code Submission Grader & Reviewer - ML Module

## 🚀 Overview

This project implements the core Machine Learning (ML) module for an automated system designed to predict verdicts and provide feedback on programming code submissions. Given a problem description (statement, input/output specifications), a student's source code, and the programming language, this module predicts the likely outcome (e.g., Accepted, Wrong Answer, Time Limit Exceeded, etc.) and can generate supplementary scores and AI-driven reviews.

The system leverages Natural Language Processing (NLP) techniques to understand problem statements and Graph Neural Networks (GNNs) to analyze the structural and semantic properties of source code, offering a novel approach to automated code assessment in e-learning environments.

**Key Features:**
*   **Multi-Language Support:** Designed to process submissions in C++, Python, and Java (with distinct models per language).
*   **Hybrid Feature Extraction:** Combines TF-IDF based text embeddings for problem descriptions with GNN-based embeddings for Abstract Syntax Trees (ASTs) of source code.
*   **Verdict Prediction:** Predicts one of seven standard programming contest verdicts.
*   **Custom Scoring:** Calculates derived "correctness" and "efficiency" scores based on verdict probabilities.
*   **AI-Generated Reviews (Optional):** Integrates with a Large Language Model (LLM) like Google's Gemini to provide human-understandable feedback on submissions.
*   **Modular Design:** Built with distinct, reusable Python components for each stage of the pipeline.
*   **API for Inference:** Includes a Flask-based web API to serve the trained models and make predictions.
*   **Analysis & Evaluation Scripts:** Provides scripts for training, evaluating models, and analyzing specific prediction cases.

## 📂 Project Structure

A brief overview of the key directories and files:
```
.
├── data/ # Placeholder for your CSV datasets (not in repo usually)
│ ├── final_problem_statements.csv
│ ├── submissions_Python.csv
│ ├── submission_stats_Python.csv
│ ├── submissions_Java.csv
│ ├── submission_stats_Java.csv
│ ├── submissions_Cpp.csv
│ └── submission_stats_Cpp.csv
├── saved_model_components/ # Where trained models and components are saved
│ ├── global_text_embedder.joblib
│ ├── python/
│ │ ├── submission_predictor_python.pth
│ │ ├── code_gnn_vocab_python.json
│ │ └── training_checkpoint_python.pth
│ ├── java/
│ │ └── ... (similar files for Java)
│ └── cpp/
│ └── ... (similar files for Cpp)
├── training_logs/ # Where training logs are saved
│ ├── training_log_python.txt
│ └── ... (logs for other languages)
├── services/ # For external services like LLM clients
│ └── gemini_client.py
├── api_config.py # Configuration for the Flask API
├── api.py # Flask application for serving the model
├── train_model.py # Script for training the models
├── evaluate_model.py # Script for evaluating trained models
├── analyze_model_predictions.py # Script for collecting specific correct/incorrect samples
├── collect_showcase_samples.py # Script to gather original files for showcase
├── extract.py # Script for extracting and sampling data from CodeNet (or similar)
├── preprocessor.py # Text preprocessing utilities
├── code_normalizer.py # Source code normalization utilities
├── code_parser.py # AST generation using Tree-sitter
├── text_embedder.py # TF-IDF based text embedding
├── code_embedder_gnn.py # GNN based code embedding
├── concat_embedder.py # Module to concatenate text and code embeddings
├── submission_dataset.py # PyTorch Dataset class
├── submission_predictor.py # The main PyTorch model for verdict prediction
├── requirements.txt # Python package dependencies
└── README.md # This file
```


## ⚙️ Core ML Module Components

The heart of this project lies in its ML pipeline, composed of the following key Python classes:

1.  **`Preprocessor` (`preprocessor.py`):** Cleans and standardizes problem statement text (statement, input/output specs) using techniques like lowercasing, tokenization, stop-word removal, and stemming.
2.  **`CodeNormalizer` (`code_normalizer.py`):** Prepares source code by removing comments and normalizing whitespace for various languages.
3.  **`CodeParser` (`code_parser.py`):** Parses normalized source code into an Abstract Syntax Tree (AST) dictionary using the Tree-sitter library, providing a structural representation of the code.
4.  **`TextEmbedder` (`text_embedder.py`):** Converts preprocessed problem statements into numerical TF-IDF vectors. A single, globally fitted `TextEmbedder` instance is typically used.
5.  **`CodeEmbedderGNN` (`code_embedder_gnn.py`):** Generates a vector embedding for a source code snippet. It internally uses the `CodeParser` to get an AST, converts this AST into a graph, and processes it using Graph Convolutional Network (GCN) layers. It maintains a language-specific vocabulary of AST node types.
6.  **`ConcatEmbedder` (`concat_embedder.py`):** Takes the vector from `TextEmbedder` and the vector from `CodeEmbedderGNN`, optionally projects them to different dimensions, and concatenates them into a single feature vector.
7.  **`SubmissionDataset` (`submission_dataset.py`):** A PyTorch `Dataset` class that loads data from CSVs, uses the `Preprocessor` on statements, normalizes code with `CodeNormalizer`, and provides items (raw/normalized code strings, preprocessed statement strings, language, and true verdict labels) for training and evaluation.
8.  **`SubmissionPredictor` (`submission_predictor.py`):** The main PyTorch `nn.Module`. It contains the `ConcatEmbedder`. Its `forward` pass takes code, statement, and language, gets the combined feature vector from the `ConcatEmbedder`, passes it through an MLP, and finally through a classification head to output logits for the 7 submission verdicts.

**Interconnections:**
The `SubmissionDataset` prepares inputs for the `SubmissionPredictor`. Inside the `SubmissionPredictor`, the `ConcatEmbedder` orchestrates calls to the `TextEmbedder` and `CodeEmbedderGNN` (which in turn uses the `CodeParser`). The `Preprocessor` and `CodeNormalizer` are used by `SubmissionDataset` to prepare the raw data before it even reaches the embedding stages.

## 🛠️ Setup & Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/eulmlk/neurograde_ml_module.git
    cd neurograde_ml_module
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include:
    ```
    pandas
    joblib
    torch
    torchvision
    torchaudio
    # Add specific torch_geometric install command here if needed, or provide link
    # e.g., torch-scatter, torch-sparse, torch-cluster, torch-spline-conv (if PyG needs them explicitly)
    torch_geometric 
    scikit-learn
    nltk
    tree_sitter
    tree_sitter_language_pack # For CodeParser
    flask # For the API
    python-dotenv # For API key management
    langchain-google-genai # For LLM integration
    tqdm # For progress bars
    matplotlib # For plotting confusion matrix
    seaborn # For plotting confusion matrix
    ```
    *   **Note on PyTorch/PyTorch Geometric:** Ensure these are compatible and installed correctly for your system (CPU or specific CUDA version). Refer to their official installation guides if the general pip install fails. For Colab, specific pip install commands with index URLs might be needed.

4.  **Download NLTK Data (First time running `Preprocessor`):**
    The `Preprocessor` might require NLTK resources. The script may attempt to download them, or you can do it manually in Python:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet') # If using lemmatization
    nltk.download('omw-1.4') # For WordNet
    ```

5.  **Set up Environment Variables (for API):**
    If using the LLM review feature in the API, create a `.env` file in the root directory (or where `services/gemini_client.py` expects it) with your Google API key:
    ```
    GOOGLE_API_KEY=your_actual_google_api_key_here
    ```

## 💾 Data Preparation (`extract.py`)

This project assumes input data is derived from a source like Project CodeNet.
*   The `extract.py` script is responsible for processing the raw CodeNet dataset (metadata and source code files) into the CSV format expected by `SubmissionDataset` (`final_problem_statements.csv`, `submissions_{Lang}.csv`, `submission_stats_{Lang}.csv`).
*   **Configuration:**
    *   Update `PROJECT_CODENET_BASE_DIR` in `extract.py` to point to your local CodeNet dataset.
    *   Adjust `GLOBAL_SAMPLING_TARGETS_PER_LANG` and `MAX_TOTAL_SAMPLES_PER_LANGUAGE` to control the data extraction and balancing strategy.
*   **Running:**
    ```bash
    python extract.py
    ```
    This will generate the CSV files in the `data/` directory (or as configured in `OUTPUT_DIR`).

## 🚂 Model Training (`train_model.py`)

The `train_model.py` script handles training a `SubmissionPredictor` model for a **single specified language** at a time. It supports resuming training and saves model components.

1.  **Configuration:** Open `train_model.py` and configure:
    *   `TARGET_LANG`: Set to `"python"`, `"java"`, or `"cpp"` for the language you want to train.
    *   `ACTUAL_..._CSV` paths: Ensure these point to the CSVs generated by `extract.py`.
    *   Hyperparameters: `TEXT_EMB_MAX_FEATURES`, `CODE_GNN_...` dimensions, `PREDICTOR_MLP_HIDDEN_DIMS`, `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`. Adjust these based on your dataset size and available hardware.
    *   `DEVICE`: Automatically detects CUDA but defaults to CPU. For Google Colab, ensure your runtime is set to GPU.
    *   `SAVE_COMPONENTS_BASE_DIR` and `LOG_DIR`: Define where models and logs are saved.
2.  **Running Training:**
    ```bash
    python train_model.py
    ```
    *   The `train_model.py` was developed to train models for multiple languages (`cpp`, `python`, `java`) concurrently in an interleaved fashion (one epoch for C++, then one for Python, etc.). To setup:
        *   Configure `LANGUAGES_TO_TRAIN` in that script.
        *   It uses a single `MASTER_LOG_FILE`.
        *   Components are still saved per language.

## 📊 Model Evaluation (`evaluate_model.py`)

This script loads a trained model and its components for a specific language and evaluates its performance on a dataset (ideally a test or validation set).

1.  **Configuration:** Open `evaluate_model.py`:
    *   Set `TARGET_LANG` to the language of the model you want to evaluate.
    *   Ensure paths to saved components (`MODEL_FILE`, `GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE`, `CODE_GNN_VOCAB_FILE`) are correct.
    *   Set `PROBLEM_CSV_EVAL`, `SUBMISSIONS_CODE_CSV_EVAL_TPL`, `SUBMISSION_STATS_CSV_EVAL_TPL` to point to your evaluation dataset CSVs.
    *   Architectural hyperparameters (`..._EVAL`) **must match** the loaded model.
2.  **Running Evaluation:**
    ```bash
    python evaluate_model.py
    ```
    *   Output will include a classification report (precision, recall, F1-score per verdict) and a confusion matrix (printed and optionally saved as an image).

## 🧪 Analyzing Predictions

*   **`analyze_model_predictions.py`:** Loads a trained model and dataset, then iterates through samples to collect and save examples (e.g., 100 correct, 100 incorrect predictions) into CSV files for manual inspection. Configure `TARGET_LANG` and paths similarly.
*   **`collect_showcase_samples.py`:** Reads an analysis CSV (like the one from `analyze_model_predictions.py`) and copies the original problem statement HTML and submission source code files into an organized "showcase" directory for easy review. Configure CodeNet paths and the input analysis CSV path.

## 🌐 API Usage (`api.py` & `api_config.py`)

A Flask-based web API is provided to serve the trained models. It can load models for multiple languages and select the appropriate one based on the request.

1.  **Configuration (`api_config.py`):**
    *   `API_SUPPORTED_LANGUAGES`: Define which languages the API will serve (e.g., `{"python": "Python", ...}`).
    *   `SAVED_COMPONENTS_BASE_DIR_API`: Path to the base directory of saved model components.
    *   Filename templates (`MODEL_FILENAME_TPL`, etc.) and path to the `GLOBAL_TEXT_EMBEDDER_VECTORIZER_FILE`.
    *   Architectural hyperparameters (`..._API`) **must match** the loaded models.
    *   `ENABLE_LLM_REVIEW` and `LLM_REVIEW_PROMPT_TEMPLATE` for Gemini integration.
2.  **Setup `services/gemini_client.py`:**
    *   Place your Gemini LLM initialization code here.
    *   Ensure `GOOGLE_API_KEY` is set as an environment variable (e.g., in a `.env` file loaded by `gemini_client.py`).
3.  **Running the API:**
    ```bash
    python api.py
    ```
    The API will start (default: `http://0.0.0.0:5000`).
4.  **Making Predictions:** Send a POST request to the `/predict` endpoint with a JSON payload:
    ```json
    {
        "statement": "Problem description text...",
        "input_spec": "Input format...",
        "output_spec": "Output format...",
        "code_submission": "source code here...",
        "language": "python" 
    }
    ```
    The API will return the predicted verdict, probabilities, custom scores, and an LLM-generated review (if enabled).

## 💡 Potential Improvements & Future Work

*   Replace TF-IDF with transformer-based embeddings (e.g., Sentence-BERT) for problem statements.
*   Use more advanced code embeddings (e.g., CodeBERT, GraphCodeBERT, or enhance GNN with data/control flow).
*   Implement multi-task learning to predict runtime and memory alongside verdicts.
*   Fine-tune LLMs for more specialized code review generation.
*   Expand language support.
*   Develop a more sophisticated UI for interaction.
