import os
from torch.utils.data import DataLoader
from submission_dataset import SubmissionDataset


class DatasetLoader:
    def __init__(
        self,
        data_dir: str,
        concat_embedder,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """
        data_dir: folder containing all CSVs
        concat_embedder: embedding callable
        batch_size, shuffle, num_workers: DataLoader params
        """
        self.data_dir = data_dir
        self.concat_emb = concat_embedder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_dataset(self, lang: str) -> SubmissionDataset:
        stats = os.path.join(self.data_dir, f"submission_stats_{lang}.csv")
        code = os.path.join(self.data_dir, f"submissions_{lang}.csv")
        problems = os.path.join(self.data_dir, "final_problem_statements.csv")
        return SubmissionDataset(stats, code, problems, self.concat_emb)

    def get_dataloader(self, lang: str) -> DataLoader:
        ds = self.get_dataset(lang)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )


from dummy_embedder import (
    DummyEmbedder,
)  # define a stub that returns random tensors


def test_loader():
    loader = DatasetLoader(
        data_dir="./data",
        concat_embedder=DummyEmbedder(),
        batch_size=40,
        shuffle=False,
        num_workers=0,
    )
    dl = loader.get_dataloader("Cpp")
    batch = next(iter(dl))
    feats, targets = batch
    # sanity checks
    print(f"Features shape: {feats.shape}")
    v, r, m = targets
    print(f"Verdict batch: {v}")
    print(f"Runtime batch: {r}")
    print(f"Memory batch: {m}")


if __name__ == "__main__":
    test_loader()
