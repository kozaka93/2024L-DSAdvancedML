import uuid
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel


def sanitize_name(name: str) -> str:
    """Sanitize the name of the experiment."""
    sanitized_name = "".join([c for c in name if c.isupper()])
    if len(sanitized_name) < 2:
        sanitized_name = name[:3]
    return sanitized_name.lower()


class Experiment(BaseModel):
    classifier: Callable
    classifier_config: Dict[str, Any] = {}
    feature_selector: Callable
    feature_selector_config: Dict[str, Any] = {}
    scores: Optional[List[int]] = None
    indices: Optional[List[np.ndarray]] = None

    @cached_property
    def experiment_name(self):
        feature_selector_name = (
            self.feature_selector.__name__ if self.feature_selector else "None"
        )
        exp_name = (
            "exp_"
            + sanitize_name(self.classifier.__name__)
            + "_"
            + sanitize_name(feature_selector_name)
        )
        if self.feature_selector:
            if (
                self.feature_selector.__name__ == "Boruta"
                and self.feature_selector_config.get("additional_feat_selector", None)
            ):
                exp_name += "_" + sanitize_name(
                    self.feature_selector_config[
                        "additional_feat_selector"
                    ].__class__.__name__
                )

            if self.feature_selector_config.get("estimator", None):
                exp_name += "_" + sanitize_name(
                    self.feature_selector_config["estimator"].__class__.__name__
                )

        exp_name += "_" + str(uuid.uuid4().hex[:6])
        return exp_name

    class Config:
        arbitrary_types_allowed = True
