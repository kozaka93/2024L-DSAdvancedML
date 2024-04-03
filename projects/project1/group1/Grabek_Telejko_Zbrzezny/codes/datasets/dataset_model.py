"""
dataset_model.py

Common dataset model class for datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd


class Dataset:
    """Common class for datasets."""

    def __init__(
        self,
        name: str,
        filename: Path,
        target_colname: str,
        load_on_import: bool = True,
    ) -> None:
        self.name = name
        self.filename = filename
        self.target_colname = target_colname
        self.df: Optional[pd.DataFrame] = None
        # used for functions applied after train-test split e.g. imputation
        self.additional_preprocess: Optional[Callable] = None
        if load_on_import:
            self.load_and_preprocess()

    def __str__(self) -> str:
        return f"{self.name.title()}{self.df.shape}"

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads the dataset and preprocesses it, sets the df attribute."""
        raise NotImplementedError("Subclasses must implement this method.")
