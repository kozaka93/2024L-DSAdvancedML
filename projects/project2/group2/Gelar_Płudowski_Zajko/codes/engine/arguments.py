from argparse import ArgumentParser

from . import AVAILABLE_OBJECTIVES


def get_shell_hpo_arguments():
    """Parse for shell's arguments

    Returns:
        Converted shell's argments
    """
    parser = ArgumentParser()
    parser.add_argument("--timeout", help="Timeout in minutes", type=float)
    parser.add_argument(
        "--objective",
        help="Objective name",
        type=str,
        choices=AVAILABLE_OBJECTIVES,
    )
    parser.add_argument("--name", help="Name of run", type=str)
    parser.add_argument(
        "--generate-summary",
        help="If true then best model summary is generated",
        action="store_true",
    )
    return parser.parse_args()
