import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

try:
    from torch_geometric.data import Data as GraphData
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:
    GraphData = None
    GCNConv = None
    global_mean_pool = None
    print("--------------------------------------------------------------------")
    print("ERROR: PyTorch Geometric (torch_geometric) is not installed or not found.")
    print("       CodeEmbedderGNN requires PyTorch Geometric.")
    print(
        "Please install it: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )
    print("--------------------------------------------------------------------")

# --- Import your CodeParser from code_parser.py ---
try:
    from code_parser import CodeParser
except ImportError:
    print("--------------------------------------------------------------------")
    print("ERROR: Failed to import 'CodeParser' from 'code_parser.py'.")
    print("       Ensure 'code_parser.py' is in the same directory or in PYTHONPATH,")
    print(
        "       and that it has all its own dependencies (like tree_sitter_language_pack)."
    )
    print("--------------------------------------------------------------------")

    # Define a dummy CodeParser if import fails, so the script can be loaded but won't work
    class CodeParser:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.parsers = {}

        def get_ast_dict(self, *args, **kwargs):
            return None

        def get_supported_languages(self, *args, **kwargs):
            return []


# --- End of CodeParser import ---


class CodeEmbedderGNN(nn.Module):
    """
    A GNN-based code embedder that parses source code into an AST,
    converts the AST into a graph, and runs it through GCNConv layers
    to produce an embedding vector.
    """

    def __init__(
        self,
        node_vocab_size: int,
        node_embedding_dim: int,
        hidden_gnn_dim: int,
        out_graph_embedding_dim: int,
        num_gnn_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if GCNConv is None or global_mean_pool is None or GraphData is None:
            raise ImportError(
                "PyTorch Geometric components not loaded. CodeEmbedderGNN cannot be initialized."
            )

        self.code_parser = CodeParser()  # Instantiates CodeParser from code_parser.py

        self.node_type_to_id: Dict[str, int] = {"<UNK_TYPE>": 0}
        self.next_node_type_id: int = 1
        self.node_vocab_size = node_vocab_size

        self.node_type_embedding = nn.Embedding(
            num_embeddings=node_vocab_size, embedding_dim=node_embedding_dim
        )

        self.convs = nn.ModuleList()
        current_dim = node_embedding_dim
        for i in range(num_gnn_layers):
            next_dim = (
                hidden_gnn_dim if i < num_gnn_layers - 1 else out_graph_embedding_dim
            )
            self.convs.append(GCNConv(current_dim, next_dim))
            current_dim = next_dim

        self.dropout = nn.Dropout(p=dropout_rate)
        self.out_dim = out_graph_embedding_dim
        self.num_node_types = node_vocab_size

    def _get_or_create_node_type_id(self, node_type_str: str) -> int:
        if node_type_str not in self.node_type_to_id:
            if self.next_node_type_id < self.node_vocab_size:
                self.node_type_to_id[node_type_str] = self.next_node_type_id
                self.next_node_type_id += 1
            else:
                return self.node_type_to_id["<UNK_TYPE>"]
        return self.node_type_to_id[node_type_str]

    def _build_graph_from_ast_dict_recursive(
        self,
        ast_node_dict: Dict[str, Any],
        node_ids_list: List[int],
        edges_list: List[List[int]],
        parent_graph_idx: Optional[int] = None,
    ) -> int:
        node_type = ast_node_dict.get("type", "<MISSING_TYPE>")
        current_node_type_id = self._get_or_create_node_type_id(node_type)
        node_ids_list.append(current_node_type_id)
        current_graph_idx = len(node_ids_list) - 1

        if parent_graph_idx is not None:
            edges_list.append([parent_graph_idx, current_graph_idx])

        for child_ast_node_dict in ast_node_dict.get("children", []):
            self._build_graph_from_ast_dict_recursive(
                child_ast_node_dict, node_ids_list, edges_list, current_graph_idx
            )
        return current_graph_idx

    def _ast_dict_to_graph_data(
        self, ast_root_dict: Optional[Dict[str, Any]]
    ) -> Optional[GraphData]:
        if not ast_root_dict:
            return None

        node_type_ids: List[int] = []
        edge_pairs: List[List[int]] = []

        self._build_graph_from_ast_dict_recursive(
            ast_root_dict, node_type_ids, edge_pairs
        )

        if not node_type_ids:
            return None

        x_node_type_ids = torch.tensor(node_type_ids, dtype=torch.long)

        if not edge_pairs:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

        return GraphData(x_node_ids=x_node_type_ids, edge_index=edge_index)

    def forward(self, graph_data: GraphData) -> torch.Tensor:
        node_features = self.node_type_embedding(graph_data.x_node_ids)

        for i, conv_layer in enumerate(self.convs):
            node_features = conv_layer(node_features, graph_data.edge_index)
            if i < len(self.convs) - 1:
                node_features = F.relu(node_features)
                node_features = self.dropout(node_features)

        batch_vector = torch.zeros(
            node_features.size(0), dtype=torch.long, device=node_features.device
        )
        graph_embedding = global_mean_pool(node_features, batch_vector)

        return graph_embedding

    def embed(self, code_str: str, lang: str) -> Optional[torch.Tensor]:
        # Check if the CodeParser from the imported file was initialized successfully
        if (
            not self.code_parser.parsers
            or not self.code_parser.get_supported_languages()
        ):
            print(
                "Error (CodeEmbedderGNN.embed): CodeParser has no loaded parsers. Cannot embed."
            )
            print(
                "       This might be due to issues in 'code_parser.py' or its dependencies."
            )
            return None

        ast_dict = self.code_parser.get_ast_dict(code_str, lang)
        if ast_dict is None:
            return None

        graph_data = self._ast_dict_to_graph_data(ast_dict)

        if (
            graph_data is None
            or graph_data.x_node_ids is None
            or graph_data.x_node_ids.nelement() == 0
        ):
            return torch.zeros(
                1, self.out_dim
            )  # Return zero vector for empty/failed graph

        # To move data to the model's device, uncomment and adapt if using GPU:
        # device = next(self.parameters()).device
        # graph_data = graph_data.to(device)

        return self.forward(graph_data)


# --- Example Usage ---
if __name__ == "__main__":
    if GCNConv is None:
        print("PyTorch Geometric not available. Skipping CodeEmbedderGNN demo.")
    else:
        print("--- CodeEmbedderGNN Demo (with imported CodeParser) ---")

        VOCAB_SIZE = 50
        NODE_EMB_DIM = 32
        HIDDEN_GNN_DIM = 64
        OUT_EMB_DIM = 128
        NUM_GNN_LAYERS = 2
        DROPOUT = 0.2

        try:
            # First, check if CodeParser itself loads languages successfully
            # This is important because CodeEmbedderGNN relies on it.
            print("\nChecking imported CodeParser instance within CodeEmbedderGNN...")
            temp_parser_check = CodeParser()
            if not temp_parser_check.get_supported_languages():
                print("CRITICAL: The imported CodeParser failed to load any languages.")
                print(
                    "          Please check 'code_parser.py' and its dependencies (like tree_sitter_language_pack)."
                )
                print(
                    "          CodeEmbedderGNN will not function correctly. Exiting demo."
                )
            else:
                print(
                    f"Imported CodeParser seems OK. Supported languages: {temp_parser_check.get_supported_languages()}"
                )

                code_embedder = CodeEmbedderGNN(
                    node_vocab_size=VOCAB_SIZE,
                    node_embedding_dim=NODE_EMB_DIM,
                    hidden_gnn_dim=HIDDEN_GNN_DIM,
                    out_graph_embedding_dim=OUT_EMB_DIM,
                    num_gnn_layers=NUM_GNN_LAYERS,
                    dropout_rate=DROPOUT,
                )
                print("\nCodeEmbedderGNN initialized.")

                sample_python_code = "def greet(name):\n    print(f'Hello, {name}!')\n"
                sample_java_code = (
                    "class Simple { void main() { System.out.println(); } }"
                )

                if "python" in temp_parser_check.get_supported_languages():
                    print(f"\nEmbedding Python code: '{sample_python_code.strip()}'")
                    py_embedding = code_embedder.embed(sample_python_code, "python")
                    if py_embedding is not None:
                        print(f"  Python embedding shape: {py_embedding.shape}")
                    else:
                        print("  Failed to get Python embedding.")
                else:
                    print(
                        "\nSkipping Python embedding test: Python parser not loaded by CodeParser."
                    )

                if "java" in temp_parser_check.get_supported_languages():
                    print(f"\nEmbedding Java code: '{sample_java_code.strip()}'")
                    java_embedding = code_embedder.embed(sample_java_code, "java")
                    if java_embedding is not None:
                        print(f"  Java embedding shape: {java_embedding.shape}")
                    else:
                        print("  Failed to get Java embedding.")
                else:
                    print(
                        "\nSkipping Java embedding test: Java parser not loaded by CodeParser."
                    )

        except ImportError as e:
            print(f"Demo skipped due to missing import: {e}")
        except Exception as e:
            print(f"An error occurred during the demo: {e}")
            import traceback

            traceback.print_exc()
