from dataclasses import dataclass, Field
from merge_args import merge_args


@dataclass
class Config:
    a: int
    b: str

class Model:
    def __init__(self, **kwargs):
        print(kwargs)

Config(a=1, b=Model())