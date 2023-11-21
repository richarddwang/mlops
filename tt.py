from merge_args import merge_args
from lightning import Trainer, Callback
from typing import Union


from jsonargparse import ArgumentParser, CLI, ActionConfigFile

class Base:
    def __init__(self, a: int, b: str):
        """haha

        Args:
            a: fdf.
            b: qq
        """
        pass

class Derived(Base):
    def __init__(self, c: float, **kwargs):
        """haha

        Args:
            c: fddfdf.
            kwargs: fdfd
        """
        super().__init__(**kwargs)

parser = ArgumentParser()
parser.add_subclass_arguments(Base, 'callback')
parser.add_argument("--config", action=ActionConfigFile)
parser.add_class_arguments(Trainer, 'trainer')
cfg = parser.parse_args()

