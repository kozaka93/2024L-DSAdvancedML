"""
datasets module

This module contains the dataset classes, their
preprocessing functions, the function to load the datasets.
"""

from enum import Enum

from datasets.datasets import (
    Booking,
    Challenger,
    Churn,
    Dataset,
    Diabetes,
    Ionosphere,
    Jungle,
    Seeds,
    Sonar,
    Water,
)


class DATASETS(Enum):
    booking = Booking
    churn = Churn
    diabetes = Diabetes
    challenger = Challenger
    jungle = Jungle
    ionosphere = Ionosphere
    water = Water
    seeds = Seeds
    sonar = Sonar


def load_dataset(dataset: DATASETS) -> Dataset:
    """Loads the dataset(s) and preprocesses it."""
    return dataset.value()
