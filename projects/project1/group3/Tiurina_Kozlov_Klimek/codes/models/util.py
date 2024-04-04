import warnings
from typing import Optional
from typing import List
import numpy as np


class ClassMapper:
    """
    Map observations from binary target variable to desired values.
    Handles mapping both ways.
    """
    _target_classes: List[int]
    _classes: Optional[List[int]]

    def __init__(self, target_classes: List[int]):
        if len(target_classes) != 2:
            raise ValueError("Provide a list of two classes")
        self._target_classes = target_classes
        self._classes = None

    @property
    def target_classes(self):
        return self._target_classes

    def _learn_classes(self, y):
        classes = sorted(list(np.unique(y)))
        if len(classes) != 2:
            raise ValueError("y should be a vector describing a binary variable")
        self._classes = classes

    @staticmethod
    def _map(y, src_classes, target_classes):
        current_classes = sorted(list(map(int, np.unique(y))))

        if current_classes != src_classes:
            warnings.warn(f"Classes in y are not the same as source classes. "
                          f"Will attempt to map anyway. {current_classes} != {src_classes}")
            for cls in current_classes:
                if cls not in src_classes:
                    raise ValueError(f"Class {cls} not in source classes {src_classes}")

        if src_classes == target_classes:
            return y

        yy = np.copy(y)
        for i, cls in enumerate(src_classes):
            target = target_classes[i]
            yy[yy == cls] = target
        return yy

    def map_to_target(self, y):
        if self._classes is None:
            self._learn_classes(y)

        return self._map(y, self._classes, self._target_classes)

    def map_from_target(self, y):
        if self._classes is None:
            raise ValueError("Without previously mapping TO target, I don't know what to map onto.")

        return self._map(y, self._target_classes, self._classes)
