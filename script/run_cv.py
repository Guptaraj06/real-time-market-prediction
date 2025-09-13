import numpy as np
import polars as pl
from project.config import CFG
from project.data_processor import DataProcessor
from project.metrics import r2_weighted
from project.models.nn import NN
from project.pipeline import FullPipeline, PipelineCV
from project.transformers import PolarsTransformer
from tqdm.auto import tqdm

MODEL_TYPE = "gru"
START = 500
NUM = 3
LOAD_MODEL = False
REFIT = True
N_SPLITS = 2
TRAIN_SIZE = None
data_processor = DataProcessor(
    f"{MODEL_TYPE}_{NUM}.0_700_cv",
    skip_days=START,
)
df = data_processor.get_train_data()
features = data_processor.features

params_nn = {
    "model_type": "gru",
    "hidden_sizes": [500],
    "dropout_rates": [0.3],
    "hidden_sizes_linear": [500, 300],
    "dropout_rates_linear": [0.2, 0.1],
    "batch_size": 1,
    "early_stopping_patience": 1,
    "lr_refit": 0.0003,
    "lr": 0.0005,
    "epochs": 2,
    "early_stopping": True,
    "lr_patience": 10,
    "lr_factor": 0.5,
}
for SEED in range(3):
    MODEL_NAME = f"{MODEL_TYPE}_{NUM}.{SEED}_700_cv"
    print(MODEL_NAME)
    model = NN(**params_nn, random_seed=SEED)
    pipeline = FullPipeline(
        model,
        preprocessor=PolarsTransformer(features),
        run_name="full",
        name=MODEL_NAME,
        load_model=LOAD_MODEL,
        features=features,
        refit=REFIT,
        change_lr=True,
    )
    cv = PipelineCV(pipeline, n_splits=N_SPLITS, train_size=TRAIN_SIZE)
    scores = cv.fit(df, verbose=True)
