# checked

from enum import Enum


class LabellingSchema(Enum):
    IOB2 = 1
    IOBES = 2


class CharEmbeddingSchema(Enum):
    CNN = 1
    LSTM = 2


class OptimizationMethod(Enum):
    SGDWithDecreasingLR = 1
    AdaDelta = 2
    Adam = 3
