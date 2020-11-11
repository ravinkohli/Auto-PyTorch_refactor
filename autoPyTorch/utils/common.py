from typing import Iterable, NamedTuple, Type


class FitRequirement(NamedTuple):
    name: str
    supported_types: Iterable[Type]
