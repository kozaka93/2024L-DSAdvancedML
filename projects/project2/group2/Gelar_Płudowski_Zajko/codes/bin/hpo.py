import warnings

import optuna

from engine import objectives
from engine.arguments import get_shell_hpo_arguments
from engine.constants import DATA_PATH, RESULTS_PATH
from engine.data import DataProvider
from engine.results import ResultService

warnings.simplefilter("ignore")


def main() -> None:
    """
    Perform HPO using extracted meta-features. HPO is performed using
    following cube: (`M` x `hp` x `rfe` x `n_feat`) where:
        * `M` - models
        * `hp` - hyperparameters of the model
        * `rfe` - RFE model used to create features ranking
        * `n_feat` - used top features from the ranking
    """
    arguments = get_shell_hpo_arguments()
    print(vars(arguments))
    data_provider = DataProvider(DATA_PATH)
    objective = getattr(objectives, arguments.objective)(data_provider)

    study = optuna.create_study(
        study_name=arguments.name,
        storage="sqlite:///" + str(RESULTS_PATH / "hpo.db"),
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, timeout=arguments.timeout * 60)

    if arguments.generate_summary:
        output_path = RESULTS_PATH / (
            f"{arguments.objective}_{arguments.name}"
        )
        result_service = ResultService(study, objective, output_path)
        result_service.generate_results()


if __name__ == "__main__":
    main()
