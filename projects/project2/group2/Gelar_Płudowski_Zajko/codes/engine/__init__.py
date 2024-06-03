from operator import attrgetter

from .objectives import Objective

AVAILABLE_OBJECTIVES = list(
    map(attrgetter("__name__"), Objective.__subclasses__())
)
