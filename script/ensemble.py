import numpy as np
import polars as pl
from project.config import CFG
from project.data_processor import DataProcessor
from project.metrics import r2_weighted
from project.models.nn import NN
from project.pipeline import FullPipeline, PipelineCV, PipelineEnsemble
from project.transformers import PolarsTransformer
from tqdm.auto import tqdm

MODEL_NAME = "ensemble"
N_SPLITS = 2
START = 1000
MODEL_NAMES = [
    "gru_2.0_700_cv",
    "gru_2.1_700_cv",
    "gru_2.2_700_cv",
    "gru_3.0_700_cv",
    "gru_3.1_700_cv",
    "gru_3.2_700_cv",
]
WEIGHTS = np.array([1.0] * len(MODEL_NAMES)) / len(MODEL_NAMES)
REFIT_MODELS = [True] * len(MODEL_NAMES)
data_processor = DataProcessor(MODEL_NAME, skip_days=START)
df = data_processor.get_train_data()
models = []
for i, model_name in enumerate(MODEL_NAMES):
    pipeline = FullPipeline(
        NN(),
        name=model_name,
        run_name="full",
        load_model=True,
        features=None,
        refit=True,
        change_lr=False,
    )
    models.append(pipeline)
pipeline = PipelineEnsemble(models, WEIGHTS, REFIT_MODELS)
cv = PipelineCV(pipeline, n_splits=N_SPLITS)
scores = cv.fit(df, verbose=True)
