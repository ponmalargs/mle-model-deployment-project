from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features import prepare_features

TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",          # RandomForest tree storage
    "src.features.prepare_features",    # our FunctionTransformer step
]

def build_pipeline(model_params: dict) -> Pipeline:
    """Feature engineering + model in one object. Input: raw trip rows."""
    return Pipeline([
        ("features", FunctionTransformer(prepare_features)),
        ("model", RandomForestRegressor(**model_params)),
    ])