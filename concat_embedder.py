import torch
import torch.nn as nn
from typing import List, Optional, Tuple  # Added Tuple
import numpy as np

try:
    from code_embedder import CodeEmbedderGNN
except ImportError:
    print("ERROR: Failed to import 'CodeEmbedderGNN' from 'code_embedder_gnn.py'.")

    # Define a dummy class for the script to load if import fails
    class CodeEmbedderGNN(nn.Module):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.out_dim = 0

        def embed(self, *args, **kwargs):
            return torch.empty(0)


try:
    from text_embedder import TextEmbedder
except ImportError:
    print("ERROR: Failed to import 'TextEmbedder' from 'text_embedder.py'.")

    # Define a dummy class
    class TextEmbedder:  # type: ignore
        def __init__(self, *args, **kwargs):
            self._fitted = False

        def transform(self, *args, **kwargs):
            return None

        def get_feature_names(self, *args, **kwargs):
            return []

        def fit(self, *args, **kwargs):
            self._fitted = True


# --- End of imports ---


class ConcatEmbedder(nn.Module):
    """
    A class that concatenates code and text embeddings into a single vector.
    It combines code embeddings from a CodeEmbedderGNN and text embeddings
    from a TextEmbedder. Optionally applies projection to embeddings before concatenation.
    """

    def __init__(
        self,
        code_embedder: CodeEmbedderGNN,
        text_embedder: TextEmbedder,
        use_projection: bool = False,
        # Factor to scale individual embedding dimensions if projection is used
        projection_dim_scale_factor: float = 0.5,
        # Or an absolute target dimension for the combined projected embedding
        # For simplicity, let's use a scale factor for individual projections.
        min_projection_dim: int = 8,
    ):  # Minimum dimension for a projection output
        super().__init__()

        if not isinstance(code_embedder, CodeEmbedderGNN):
            raise TypeError("code_embedder must be an instance of CodeEmbedderGNN.")
        if not isinstance(text_embedder, TextEmbedder):
            raise TypeError("text_embedder must be an instance of TextEmbedder.")

        self.code_emb_module = code_embedder
        self.text_emb_module = text_embedder

        # Determine embedding dimensions
        self.code_original_dim = self.code_emb_module.out_dim

        if not self.text_emb_module._fitted:
            # This is a critical warning. If projections depend on this, dimensions will be wrong.
            # The alternative is to require `text_embedding_dim` as an __init__ argument.
            print(
                "Warning (ConcatEmbedder): TextEmbedder instance is not fitted. "
                "Text embedding dimension will be based on current (possibly empty) vocab "
                "or default to 0 if vocab is truly empty. "
                "Fit TextEmbedder before initializing ConcatEmbedder if using projection."
            )
            try:
                self.text_original_dim = len(self.text_emb_module.get_feature_names())
            except RuntimeError:  # Not fitted
                self.text_original_dim = 0
            if self.text_original_dim == 0 and use_projection:
                print(
                    "CRITICAL WARNING (ConcatEmbedder): TextEmbedder not fitted and vocab empty, "
                    "but projection is enabled. Text projection layer cannot be properly initialized."
                )

        else:
            self.text_original_dim = len(self.text_emb_module.get_feature_names())
            if self.text_original_dim == 0:
                print(
                    "Warning (ConcatEmbedder): TextEmbedder is fitted, but vocabulary is empty. "
                    "Text embedding dimension is 0."
                )

        self.use_projection = use_projection
        self.code_proj_layer: Optional[nn.Linear] = None
        self.text_proj_layer: Optional[nn.Linear] = None

        current_code_dim = self.code_original_dim
        current_text_dim = self.text_original_dim

        if self.use_projection:
            # Project code embedding
            if self.code_original_dim > 0:
                projected_code_d = max(
                    min_projection_dim,
                    int(self.code_original_dim * projection_dim_scale_factor),
                )
                self.code_proj_layer = nn.Linear(
                    self.code_original_dim, projected_code_d
                )
                current_code_dim = projected_code_d
                print(
                    f"Info (ConcatEmbedder): Code projection: {self.code_original_dim} -> {current_code_dim}"
                )
            else:
                print(
                    "Warning (ConcatEmbedder): Code original dimension is 0. No code projection layer created."
                )

            # Project text embedding
            if self.text_original_dim > 0:
                projected_text_d = max(
                    min_projection_dim,
                    int(self.text_original_dim * projection_dim_scale_factor),
                )
                self.text_proj_layer = nn.Linear(
                    self.text_original_dim, projected_text_d
                )
                current_text_dim = projected_text_d
                print(
                    f"Info (ConcatEmbedder): Text projection: {self.text_original_dim} -> {current_text_dim}"
                )
            else:
                print(
                    "Warning (ConcatEmbedder): Text original dimension is 0 (or TextEmbedder not fitted). "
                    "No text projection layer created."
                )

        self.final_dim = current_code_dim + current_text_dim
        if self.final_dim == 0:
            print(
                "Warning (ConcatEmbedder): Final concatenated dimension is 0. "
                "This likely means both code and text embeddings (or their projections) are zero-dimensional."
            )
        else:
            print(
                f"Info (ConcatEmbedder): Final concatenated embedding dimension: {self.final_dim}"
            )

        # For compatibility with original class attribute names:
        self.code_emb = self.code_emb_module  # Expose original attribute name
        self.text_emb = self.text_emb_module  # Expose original attribute name

    def forward(
        self,
        code_list: List[str],
        text_list: List[str],  # Assumed to be preprocessed for TextEmbedder
        lang: str,
    ) -> torch.Tensor:
        """
        Embeds a batch of code strings and text strings, then concatenates them.

        Args:
            code_list: List of source code strings.
            text_list: List of preprocessed problem statement strings.
            lang: The programming language of the code.

        Returns:
            A PyTorch Tensor of concatenated embeddings, shape (batch_size, self.final_dim).
        """
        if not code_list and not text_list:
            # Return an empty tensor with the correct feature dimension if inputs are empty
            return torch.empty(0, self.final_dim, device=self._get_module_device())

        if len(code_list) != len(text_list):
            raise ValueError(
                "Code list and text list must have the same number of items (batch size)."
            )

        batch_size = len(code_list)

        # 1. Get Code Embeddings
        # CodeEmbedderGNN.embed processes one code string at a time.
        # We need to loop and stack, or modify CodeEmbedderGNN to process batches if feasible.
        # For now, loop and stack.
        code_embeddings_list = []
        for code_str in code_list:
            # The embed method should handle moving its internal graph_data to the model's device
            code_emb = self.code_emb_module.embed(code_str, lang)
            if code_emb is None:  # Parsing or embedding failed
                # Return a zero tensor of the expected original code dimension
                code_emb = torch.zeros(
                    1, self.code_original_dim, device=self._get_module_device()
                )
            code_embeddings_list.append(code_emb)

        if code_embeddings_list:
            batch_code_embeddings = torch.cat(
                code_embeddings_list, dim=0
            )  # (batch_size, code_original_dim)
        else:  # Should not happen if batch_size > 0 due to earlier checks
            batch_code_embeddings = torch.empty(
                0, self.code_original_dim, device=self._get_module_device()
            )

        # 2. Get Text Embeddings
        # TextEmbedder.transform works on a list of texts.
        if not self.text_emb_module._fitted:
            # This is a fallback if the warning at init was ignored.
            # A robust system might prevent forward pass or return zeros.
            print(
                "Error (ConcatEmbedder.forward): TextEmbedder is not fitted. Cannot produce text embeddings."
            )
            # Create zero tensor for text part if it has a dimension, else an empty tensor
            text_dim_for_zeros = (
                self.text_original_dim
                if self.text_original_dim > 0
                else (self.text_proj_layer.out_features if self.text_proj_layer else 0)
            )
            batch_text_embeddings_np = np.zeros((batch_size, text_dim_for_zeros))
        else:
            # TextEmbedder.transform returns a NumPy array
            batch_text_embeddings_np = self.text_emb_module.transform(text_list)
            if (
                batch_text_embeddings_np is None
            ):  # Should not happen with current TextEmbedder if fitted
                batch_text_embeddings_np = np.zeros(
                    (batch_size, self.text_original_dim)
                )

        # Convert NumPy text embeddings to PyTorch tensor
        batch_text_embeddings = torch.tensor(
            batch_text_embeddings_np,
            dtype=torch.float32,
            device=self._get_module_device(),
        )

        # 3. Apply Projections (if enabled)
        projected_code_embeddings = batch_code_embeddings
        if self.code_proj_layer:
            projected_code_embeddings = self.code_proj_layer(batch_code_embeddings)

        projected_text_embeddings = batch_text_embeddings
        if self.text_proj_layer:
            projected_text_embeddings = self.text_proj_layer(batch_text_embeddings)

        # 4. Concatenate
        # Ensure dimensions are as expected after projection (or original if no projection)
        # Handle cases where one of the embeddings might be zero-dimensional effectively

        embeddings_to_concat = []
        if projected_code_embeddings.shape[1] > 0:
            embeddings_to_concat.append(projected_code_embeddings)
        elif self.final_dim > 0 and (
            self.code_original_dim > 0
            or (self.code_proj_layer and self.code_proj_layer.out_features > 0)
        ):
            # If code part was supposed to have a dimension but is now empty, fill with zeros
            code_target_dim = (
                self.code_proj_layer.out_features
                if self.code_proj_layer
                else self.code_original_dim
            )
            embeddings_to_concat.append(
                torch.zeros(
                    batch_size, code_target_dim, device=self._get_module_device()
                )
            )

        if projected_text_embeddings.shape[1] > 0:
            embeddings_to_concat.append(projected_text_embeddings)
        elif self.final_dim > 0 and (
            self.text_original_dim > 0
            or (self.text_proj_layer and self.text_proj_layer.out_features > 0)
        ):
            text_target_dim = (
                self.text_proj_layer.out_features
                if self.text_proj_layer
                else self.text_original_dim
            )
            embeddings_to_concat.append(
                torch.zeros(
                    batch_size, text_target_dim, device=self._get_module_device()
                )
            )

        if not embeddings_to_concat:
            # Both embeddings are effectively zero-dimensional
            return torch.empty(batch_size, 0, device=self._get_module_device())

        concatenated_embeddings = torch.cat(embeddings_to_concat, dim=1)

        # Final check for dimension consistency
        if concatenated_embeddings.shape[1] != self.final_dim:
            # This might happen if one of the embedders dynamically changed output dim or projection was misconfigured
            print(
                f"Warning (ConcatEmbedder.forward): Actual concatenated dim ({concatenated_embeddings.shape[1]}) "
                f"differs from expected final_dim ({self.final_dim}). This could indicate an issue."
            )
            # Fallback: create zero tensor of expected final_dim if shape is wrong but final_dim is positive
            if (
                self.final_dim > 0 and concatenated_embeddings.shape[0] == batch_size
            ):  # if batch size is ok
                concatenated_embeddings = torch.zeros(
                    batch_size, self.final_dim, device=self._get_module_device()
                )

        return concatenated_embeddings

    def _get_module_device(self) -> torch.device:
        """Helper to get the device of the module's parameters."""
        # If module has no parameters (e.g. only contains other modules that might be empty),
        # default to CPU. This can happen if projections are not used and embedders are on CPU.
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


# --- Example Usage ---
if __name__ == "__main__":
    print("--- ConcatEmbedder Demo ---")

    # Dummy data for testing
    sample_corpus_text = [
        "this is the first document statement for problem one",
        "problem two has this statement text for document",
        "and the third problem statement",
        "final problem with its own unique text document",
    ]
    sample_codes_py = [
        "def solve_one():\n  print('problem one solution')\n",
        "def solve_two(n):\n  return n * n\n",
        "print('solution for three')",
        "class Solver:\n  def run(self):\n    pass\n",
    ]
    sample_lang = "python"
    batch_size = len(sample_codes_py)

    # 1. Initialize and Fit TextEmbedder
    print("\n1. Initializing and fitting TextEmbedder...")
    try:
        # Using defaults from text_embedder.py (max_features=5000)
        # For demo, let's use a smaller max_features
        text_embedder_instance = TextEmbedder(max_features=20)
        text_embedder_instance.fit(sample_corpus_text)
        print(
            f"   TextEmbedder fitted. Vocab size: {len(text_embedder_instance.get_feature_names())}"
        )
    except Exception as e:
        print(f"   Error initializing/fitting TextEmbedder: {e}")
        text_embedder_instance = None  # type: ignore

    # 2. Initialize CodeEmbedderGNN
    print("\n2. Initializing CodeEmbedderGNN...")
    try:
        # Match these with CodeEmbedderGNN's __init__
        code_embedder_instance = CodeEmbedderGNN(
            node_vocab_size=50,
            node_embedding_dim=16,  # smaller for demo
            hidden_gnn_dim=32,
            out_graph_embedding_dim=24,  # smaller for demo
            num_gnn_layers=1,  # simpler for demo
        )
        print(
            f"   CodeEmbedderGNN initialized. Output dim: {code_embedder_instance.out_dim}"
        )
    except Exception as e:
        print(f"   Error initializing CodeEmbedderGNN: {e}")
        code_embedder_instance = None  # type: ignore

    if text_embedder_instance and code_embedder_instance:
        # 3. Initialize ConcatEmbedder
        print("\n3. Initializing ConcatEmbedder (no projection)...")
        try:
            concat_embedder_no_proj = ConcatEmbedder(
                code_embedder=code_embedder_instance,
                text_embedder=text_embedder_instance,
                use_projection=False,
            )
            print(
                f"   ConcatEmbedder (no proj) initialized. Final dim: {concat_embedder_no_proj.final_dim}"
            )

            # Test forward pass
            print("   Testing forward pass (no projection)...")
            # Ensure model and data are on the same device if testing on GPU
            # concat_embedder_no_proj.to(device) # if device is defined
            concatenated_output = concat_embedder_no_proj(
                sample_codes_py, sample_corpus_text, sample_lang
            )
            print(
                f"     Output shape: {concatenated_output.shape}"
            )  # Expected: (batch_size, final_dim)
            if (
                concatenated_output.shape[0] == batch_size
                and concatenated_output.shape[1] == concat_embedder_no_proj.final_dim
            ):
                print("     Forward pass shape is correct.")
            else:
                print("     WARNING: Forward pass shape MISMATCH.")

        except Exception as e:
            print(f"   Error initializing or using ConcatEmbedder (no proj): {e}")
            import traceback

            traceback.print_exc()

        print("\n4. Initializing ConcatEmbedder (with projection)...")
        try:
            concat_embedder_with_proj = ConcatEmbedder(
                code_embedder=code_embedder_instance,
                text_embedder=text_embedder_instance,
                use_projection=True,
                projection_dim_scale_factor=0.5,  # Project to 50%
                min_projection_dim=4,  # Ensure projected dim is at least 4
            )
            print(
                f"   ConcatEmbedder (with proj) initialized. Final dim: {concat_embedder_with_proj.final_dim}"
            )

            # Test forward pass
            print("   Testing forward pass (with projection)...")
            concatenated_output_proj = concat_embedder_with_proj(
                sample_codes_py, sample_corpus_text, sample_lang
            )
            print(f"     Output shape: {concatenated_output_proj.shape}")
            if (
                concatenated_output_proj.shape[0] == batch_size
                and concatenated_output_proj.shape[1]
                == concat_embedder_with_proj.final_dim
            ):
                print("     Forward pass shape is correct.")
            else:
                print("     WARNING: Forward pass shape MISMATCH.")

        except Exception as e:
            print(f"   Error initializing or using ConcatEmbedder (with proj): {e}")
            import traceback

            traceback.print_exc()

    else:
        print(
            "\nSkipping ConcatEmbedder tests due to errors in initializing prior embedders."
        )

    print("\n--- ConcatEmbedder Demo Finished ---")
