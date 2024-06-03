import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm
from sklearn.base import clone
from utils import calculate_score

class IterativeSelector:

    def __init__(
        self,
        models: list,
        scoring_functions: list = [calculate_score],
        direction: str = "forward",
    ):
        """
        This class was designed with the custom scoring function for this task in mind.
        However, it doesn't need to be the main scoring functions. The order of scoring functions in the list matters in feature selection.
        """
        self.models = models
        self.scoring_functions = scoring_functions

        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be either 'forward' or 'backward'")
        self.direction = direction

    def __append_score(self, scores_step, model, scoring_function, features, score):
        scores_step.append({
            "model": model.__str__(),
            "scoring_function": scoring_function.__name__,
            "features": ",".join(features),
            "score": score,
        })

    def __get_score(
        self, scoring_function, model, predictions, X_test, y_test, features
    ):
        """
            Method for getting the scores from different scoring functions.
        """
        if (scoring_function.__name__ == "precision_calculate_score"): 
            return scoring_function(X_test[features], y_test, model)
        if (scoring_function.__name__ == "calculate_score"): 
            return scoring_function(X_test[features], y_test, model)
        else:
            return scoring_function(y_test, predictions)

    def __is_better(self, new_best):
        """
            Comparison of global best vs best of this step
        """
        for metric in self.scoring_functions:
            metric = metric.__name__
            new_value = new_best.get(metric)
            old_value = self.best_scores.get(metric)

            if new_value > old_value:
                return True  # replace old dict with new dict
            elif new_value < old_value:
                return False  # keep the old dict
            # if values are equal, continue to the next metric 
        return True  # If all values are equal, keep the old dict

    def __update_variables(
        self, scores_step, steps_since_improvement, current_features, remaining_features
    ):
        """
            At the end of each step. Updates selector varaibles.
        """
        df = pd.DataFrame(scores_step)
        best = (
            df.pivot_table(
                index=["model", "features"], columns="scoring_function", values="score"
            )  # long to wide
            .reset_index()
            .sort_values(
                by=[k.__name__ for k in self.scoring_functions], ascending=False
            )
            .iloc[0, :]
            .to_dict()
        )
        new_best_scores = {k.__name__: best[k.__name__] for k in self.scoring_functions}
        if self.__is_better(new_best_scores):
            self.best_scores = new_best_scores
            self.best_features = best["features"].split(",")
            steps_since_improvement = 0

        current_features = best["features"].split(",")
        if self.direction == "forward":
            remaining_features = list(set(remaining_features) - set(current_features))

        return steps_since_improvement, current_features, remaining_features

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        steps_tolerance=1,
        initial_features=[],
        remaining_features=[],
    ):
        """
            Main function. Perform iterative feature selection.
            If self.direction == forward:
                goes from initial_features to remaining_features
            If self.direction == backward:
                remaining_features is not used
                
            steps_tolerance defines how many steps without improvement of the main scoring functions are permitted
        """  
        
        current_features = initial_features.copy()
        remaining_features = remaining_features if len(remaining_features) != 0 else X_train.columns
        if self.direction == "forward" and len(current_features) != 0:
            remaining_features = list(set(remaining_features) - set(initial_features))

        self.scores = pd.DataFrame(
            columns=["model", "scoring_function", "features", "score", "n_features"]
        )
        self.best_scores = {f.__name__: -np.inf for f in self.scoring_functions}
        self.best_features = []
        steps_since_improvement = -1


        while steps_since_improvement <= steps_tolerance and (
            (self.direction == "forward" and len(remaining_features) > 0)
            or (self.direction == "backward" and len(current_features) > 1)
        ):
            steps_since_improvement += 1
            scores_step = []
            features_to_iterate = remaining_features if self.direction == "forward" else current_features
            for feature in tqdm(features_to_iterate):
                if self.direction == "forward":
                    features_to_try = current_features + [feature]
                elif self.direction == "backward":
                    features_to_try = list(set(current_features) - set([feature]))
                    
                for model in self.models:
                    model = clone(model)
                    model.fit(X_train[features_to_try], y_train)
                    predictions = model.predict(X_test[features_to_try])

                    for scoring_function in self.scoring_functions:
                        score = self.__get_score(
                            scoring_function=scoring_function,
                            model=model,
                            predictions=predictions,
                            X_test=X_test,
                            y_test=y_test,
                            features=features_to_try,
                        )
                        self.__append_score(
                            scores_step=scores_step,
                            model=model,
                            scoring_function=scoring_function,
                            features=features_to_try,
                            score=score
                        )

            steps_since_improvement, current_features, remaining_features = (
                self.__update_variables(
                    scores_step=scores_step,
                    steps_since_improvement=steps_since_improvement,
                    current_features=current_features,
                    remaining_features=remaining_features,
                )
            )
            
            to_append = pd.DataFrame(scores_step)
            to_append["n_features"] = len(features_to_try)
            self.scores = pd.concat((self.scores, to_append))

            if steps_since_improvement == 0:
                print(
                    "New best scores:",
                    self.best_scores,
                    "Features:",
                    self.best_features,
                )