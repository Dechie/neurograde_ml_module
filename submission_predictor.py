# submission_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # For potential use in loss function or activations
from typing import List, Tuple, Optional

# --- Import your ConcatEmbedder ---
# Ensure concat_embedder.py is in the same directory or PYTHONPATH
try:
    from concat_embedder import ConcatEmbedder
except ImportError:
    print("ERROR: Failed to import 'ConcatEmbedder' from 'concat_embedder.py'.")

    # Define a dummy class for the script to load if import fails
    class ConcatEmbedder(nn.Module):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.final_dim = 0

        def forward(self, *args, **kwargs):
            return torch.empty(0)

        def __call__(self, *args, **kwargs):
            return torch.empty(0)


# --- End of imports ---


class SubmissionPredictor(nn.Module):
    """
    A neural network model for predicting the verdict of code submissions
    based on concatenated code and problem statement embeddings.
    """

    def __init__(
        self,
        concat_embedder: ConcatEmbedder,
        num_verdict_classes: int,  # Number of distinct verdict outcomes (e.g., 7)
        mlp_hidden_dims: Optional[
            List[int]
        ] = None,  # List of hidden layer sizes for MLP
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if not isinstance(concat_embedder, ConcatEmbedder):
            raise TypeError("concat_embedder must be an instance of ConcatEmbedder.")
        if num_verdict_classes <= 0:
            raise ValueError("num_verdict_classes must be a positive integer.")

        self.concat_embedder = concat_embedder

        # The input dimension to the MLP is the output dimension of the ConcatEmbedder
        current_dim = self.concat_embedder.final_dim
        if current_dim == 0:
            raise ValueError(
                "ConcatEmbedder's final_dim is 0. Cannot initialize SubmissionPredictor MLP."
            )

        # --- Shared MLP Layers (before the prediction head) ---
        if mlp_hidden_dims is None:
            # Default MLP: one hidden layer, half the size of concat_embedder output
            # or at least as large as num_verdict_classes if that's larger.
            mlp_hidden_dims = [max(num_verdict_classes, current_dim // 2)]
            if (
                mlp_hidden_dims[0] == 0 and current_dim > 0
            ):  # Ensure hidden dim is positive
                mlp_hidden_dims = [max(1, num_verdict_classes)]

        mlp_layers: List[nn.Module] = []
        if (
            mlp_hidden_dims and mlp_hidden_dims[0] > 0
        ):  # Only add MLP if hidden_dims are specified and positive
            for hidden_dim in mlp_hidden_dims:
                if hidden_dim <= 0:  # Skip non-positive hidden dimensions
                    print(
                        f"Warning (SubmissionPredictor): Skipping non-positive hidden_dim ({hidden_dim}) in MLP."
                    )
                    continue
                mlp_layers.append(nn.Linear(current_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout_rate))
                current_dim = hidden_dim  # Output of this layer is input to next

        if (
            not mlp_layers and current_dim != self.concat_embedder.final_dim
        ):  # If mlp_hidden_dims was empty or all non-positive
            print(
                "Info (SubmissionPredictor): No positive MLP hidden layers specified. "
                "Verdict head will connect directly to ConcatEmbedder output."
            )
            current_dim = (
                self.concat_embedder.final_dim
            )  # Reset current_dim to concat output
        elif mlp_layers:
            print(
                f"Info (SubmissionPredictor): MLP created with hidden dimensions: {mlp_hidden_dims}. "
                f"Input to verdict head: {current_dim}"
            )
        else:  # No MLP layers, current_dim is still concat_embedder.final_dim
            print(
                f"Info (SubmissionPredictor): No MLP hidden layers. "
                f"Input to verdict head: {current_dim}"
            )

        self.shared_mlp = nn.Sequential(*mlp_layers)

        # --- Prediction Head for Verdicts ---
        # This layer outputs raw scores (logits) for each verdict class.
        # Softmax will be applied later (usually in the loss function like CrossEntropyLoss).
        self.verdict_head = nn.Linear(current_dim, num_verdict_classes)

        # For compatibility with original problem description attributes (though only encoder is directly used)
        self.encoder = self.concat_embedder  # alias

        print(
            f"SubmissionPredictor initialized. Input features: {self.concat_embedder.final_dim}, "
            f"Output verdict classes: {num_verdict_classes}"
        )

    def forward(
        self,
        code_list: List[str],
        text_list: List[
            str
        ],  # Assumed preprocessed for TextEmbedder within ConcatEmbedder
        lang: str,
    ) -> torch.Tensor:  # Returns verdict logits
        """
        Performs the forward pass of the model.

        Args:
            code_list: List of source code strings.
            text_list: List of preprocessed problem statement strings.
            lang: The programming language of the code.

        Returns:
            torch.Tensor: Raw output logits for verdict classification,
                          shape (batch_size, num_verdict_classes).
        """
        # 1. Get concatenated embeddings from the encoder (ConcatEmbedder)
        # The ConcatEmbedder's forward method is called here.
        # It handles code embedding (via CodeEmbedderGNN -> CodeParser)
        # and text embedding (via TextEmbedder).
        # x will have shape (batch_size, self.concat_embedder.final_dim)
        x = self.concat_embedder(code_list, text_list, lang)

        # 2. Pass through the shared MLP layers (if any)
        x_processed = self.shared_mlp(x)

        # 3. Get verdict predictions (logits)
        verdict_logits = self.verdict_head(x_processed)

        return verdict_logits


if __name__ == "__main__":
    print("--- SubmissionPredictor Demo ---")

    # Ensure dummy/mock or real versions of dependencies are available for demo
    # For this demo, we'll need to initialize the full chain:
    # Preprocessor, CodeNormalizer, TextEmbedder, CodeParser (inside CodeEmbedderGNN),
    # CodeEmbedderGNN, and ConcatEmbedder.

    # Import necessary classes for the demo setup
    # These should be actual imports if running in a project structure
    from text_embedder import TextEmbedder
    from code_embedder import CodeEmbedderGNN

    # CodeParser is used internally by CodeEmbedderGNN
    # Preprocessor and CodeNormalizer would be used by SubmissionDataset if it were part of this demo's focus,
    # but SubmissionPredictor only needs the ConcatEmbedder.

    # For simplicity, we'll re-use parts of ConcatEmbedder's demo setup.

    # A. Setup for TextEmbedder
    sample_corpus_text_for_fitting = [
        "problem one text",
        "problem two different text",
        "statement for three",
    ]
    try:
        text_embedder_instance = TextEmbedder(
            max_features=15
        )  # Keep vocab small for demo
        # Crucially, FIT the TextEmbedder
        # In a real pipeline, preprocessor would be used here. For this isolated demo, assume text is "preprocessed".
        text_embedder_instance.fit(sample_corpus_text_for_fitting)
        print(
            f"TextEmbedder fitted. Vocab size: {len(text_embedder_instance.get_feature_names())}"
        )
        TEXT_EMB_DIM = len(text_embedder_instance.get_feature_names())
    except Exception as e:
        print(
            f"Error setting up TextEmbedder for demo: {e}. SubmissionPredictor demo might fail."
        )

        # Provide a dummy if setup fails, to allow ConcatEmbedder to init (though it will warn)
        class DummyTextEmbedder:
            _fitted = False

            def get_feature_names(self):
                return []

        text_embedder_instance = DummyTextEmbedder()  # type: ignore
        TEXT_EMB_DIM = 0

    # B. Setup for CodeEmbedderGNN
    try:
        # These dimensions are examples
        CODE_GNN_OUT_DIM = 24
        code_embedder_gnn_instance = CodeEmbedderGNN(
            node_vocab_size=30,
            node_embedding_dim=10,
            hidden_gnn_dim=16,
            out_graph_embedding_dim=CODE_GNN_OUT_DIM,
            num_gnn_layers=1,
        )
        print(
            f"CodeEmbedderGNN initialized. Output dim: {code_embedder_gnn_instance.out_dim}"
        )
    except Exception as e:
        print(
            f"Error setting up CodeEmbedderGNN for demo: {e}. SubmissionPredictor demo might fail."
        )

        # Provide a dummy if setup fails
        class DummyCodeGNN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.out_dim = 0

            def embed(self, *args, **kwargs):
                return torch.zeros(1, self.out_dim)

        code_embedder_gnn_instance = DummyCodeGNN()  # type: ignore
        CODE_GNN_OUT_DIM = 0

    # C. Setup for ConcatEmbedder (the input to SubmissionPredictor)
    try:
        concat_embedder_instance = ConcatEmbedder(
            code_embedder=code_embedder_gnn_instance,
            text_embedder=text_embedder_instance,  # Pass the FITTED text_embedder
            use_projection=False,  # Keep it simple for this demo
        )
        print(
            f"ConcatEmbedder initialized. Final output dim: {concat_embedder_instance.final_dim}"
        )
    except Exception as e:
        print(
            f"Error setting up ConcatEmbedder for demo: {e}. SubmissionPredictor demo might fail."
        )

        # Provide a dummy ConcatEmbedder if previous steps failed, so Predictor can attempt init
        class DummyConcat(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.final_dim = TEXT_EMB_DIM + CODE_GNN_OUT_DIM

            def forward(self, *args, **kwargs):
                return torch.zeros(
                    args[0].__len__(), self.final_dim
                )  # args[0] is code_list

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        concat_embedder_instance = DummyConcat()  # type: ignore

    if concat_embedder_instance.final_dim == 0:
        print(
            "CRITICAL: ConcatEmbedder final_dim is 0. Predictor cannot be properly initialized. Demo aborted."
        )
    else:
        # D. Initialize SubmissionPredictor
        NUM_VERDICT_CLASSES = 7  # Based on your updated list
        MLP_LAYERS = [64, 32]  # Example MLP hidden layers

        print("\nInitializing SubmissionPredictor...")
        try:
            predictor_model = SubmissionPredictor(
                concat_embedder=concat_embedder_instance,
                num_verdict_classes=NUM_VERDICT_CLASSES,
                mlp_hidden_dims=MLP_LAYERS,
                dropout_rate=0.25,
            )
            print("SubmissionPredictor model initialized:")
            print(predictor_model)  # Prints the model structure

            # E. Prepare dummy batch input for a forward pass
            batch_size = 2
            dummy_code_list = [
                "def main():\n  print('hello') # Python code",
                "public class Solution { public void run() {} } // Java code",
            ]
            # For TextEmbedder, these texts should ideally be "preprocessed"
            # In a real pipeline, the SubmissionDataset handles preprocessing.
            # Here, we pass them as is, and our dummy TextEmbedder (if used) or real one handles it.
            dummy_text_list = [
                "problem statement for first code sample",
                "description for the second submission",
            ]
            dummy_lang_py = "python"  # Assuming CodeEmbedderGNN can handle python
            dummy_lang_java = (
                "java"  # Assuming CodeEmbedderGNN can handle java (if parser available)
            )

            print("\nTesting forward pass with a batch (Python)...")
            # Model expects lists of strings for code and text
            verdict_logits_py = predictor_model(
                dummy_code_list, dummy_text_list, dummy_lang_py
            )
            print(
                f"  Verdict logits shape (Python): {verdict_logits_py.shape}"
            )  # Expected: (batch_size, NUM_VERDICT_CLASSES)
            print(
                f"  Sample logits (Python, first item): {verdict_logits_py[0].detach().cpu().numpy()}"
            )

            # Example with a different language, if your CodeParser/CodeEmbedderGNN supports it
            # This assumes the same ConcatEmbedder instance is used, which is fine if CodeEmbedderGNN handles multiple langs.
            # print("\nTesting forward pass with a batch (Java)...")
            # verdict_logits_java = predictor_model(dummy_code_list, dummy_text_list, dummy_lang_java)
            # print(f"  Verdict logits shape (Java): {verdict_logits_java.shape}")

            # --- Example of how to get probabilities and predictions ---
            if verdict_logits_py.numel() > 0:  # Check if output is not empty
                probabilities = F.softmax(verdict_logits_py, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                print("\n  Example Probabilities (first item):")
                print(f"    {probabilities[0].detach().cpu().numpy()}")
                print(
                    f"  Example Predicted Class (first item): {predicted_classes[0].item()}"
                )

        except ValueError as ve:  # Catch ValueErrors from __init__ specifically
            print(f"ValueError during SubmissionPredictor initialization: {ve}")
        except ImportError as ie:
            print(
                f"ImportError during SubmissionPredictor demo (likely missing a component like ConcatEmbedder): {ie}"
            )
        except Exception as e:
            print(
                f"An error occurred during the SubmissionPredictor demo: {type(e).__name__} - {e}"
            )
            import traceback

            traceback.print_exc()

    print("\n--- SubmissionPredictor Demo Finished ---")
