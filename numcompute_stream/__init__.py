"""NumCompute-Stream: NumPy decision trees and incremental learning tools."""

__version__ = "0.1.0"
from .tree import DecisionTreeClassifier
from .pipeline import Pipeline
from .preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, Imputer, OneHotEncoder
from .stats import RunningStats, StreamingStats
from .metrics import StreamingMetrics
from .stream import StreamTrainer

__all__ = [
    "DecisionTreeClassifier",
    "Pipeline",
    "StandardScaler",
    "MinMaxScaler",
    "SimpleImputer",
    "Imputer",
    "OneHotEncoder",
    "RunningStats",
    "StreamingStats",
    "StreamingMetrics",
    "StreamTrainer",
]
