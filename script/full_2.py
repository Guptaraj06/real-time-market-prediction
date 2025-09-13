import numpy as np
import polars as pl
from project.config import CFG
from project.data_processor import DataProcessor
from project.metrics import r2_weighted
from project.models.nn import NN
from project.pipeline import FullPipeline
from project.transformers import PolarsTransformer
from tqdm.auto import tqdm

MODEL_TYPE = "gru"
START = 500
NUM = 2
LOAD_MODEL = False
REFIT = True
data_processor = DataProcessor(
    f"{MODEL_TYPE}_{NUM}.0_700", skip_days=START, transformer=PolarsTransformer()
)
df = data_processor.get_train_data()
features = data_processor.features

params_nn = {
    "model_type": "gru",
    "hidden_sizes": [250, 150, 150],
    "dropout_rates": [0.0, 0.0, 0.0],
    "hidden_sizes_linear": [],
    "dropout_rates_linear": [],
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
    MODEL_NAME = f"{MODEL_TYPE}_{NUM}.{SEED}_700"
    print(MODEL_NAME)
    model = NN(**params_nn, random_seed=SEED)
    df_train = df.filter(pl.col(CFG.COL_DATE) >= START + 200)
    df_valid = df.filter(pl.col(CFG.COL_DATE) < START + 200)
    pipeline = FullPipeline(
        model,
        preprocessor=None,
        run_name="full",
        name=MODEL_NAME,
        load_model=LOAD_MODEL,
        features=features,
        refit=REFIT,
    )
    pipeline.fit(df_train, df_valid, verbose=True)

    cnt_dates = 0
    preds = []
    dates = np.unique(df_valid.select(pl.col(CFG.COL_DATE)).to_series().to_numpy())
    for date_id in tqdm(dates):
        df_valid_date = df.filter(pl.col(CFG.COL_DATE) == date_id)
        if pipeline.refit & (cnt_dates > 0):
            df_valid_upd = df.filter(pl.col(CFG.COL_DATE) == date_id - 1)
            pipeline.update(df_valid_upd)
        preds_i, hidden = pipeline.predict(df_valid_date, n_times=None)
        preds += list(preds_i)
        cnt_dates += 1

    preds = np.array(preds)
    y = df_valid.select(pl.col(CFG.COL_TARGET)).to_series().to_numpy()
    weight = df_valid.select(pl.col(CFG.COL_WEIGHT)).to_series().to_numpy()
    score = r2_weighted(y, preds, weight)
    print(f"Score: {score:.5f}")
