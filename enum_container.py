from enum import Enum

class TaskType(Enum):
    css = 1
    kaggle = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return TaskType[s]
        except KeyError:
            raise ValueError()

# Level 0 Model
class DAEModelType(Enum):
    DeepStack = 1
    DeepBottleneck = 2
    TransformerAutoEncoder = 3

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DAEModelType[s]
        except KeyError:
            raise ValueError()

# Level 1 Model
class MetaModelType(Enum):
    MLP = 1
    Catboost = 2
    LGBM = 3
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return MetaModelType[s]
        except KeyError:
            raise ValueError()