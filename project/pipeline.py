import copy
import gc
import os

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from project import utils
from project.config import CFG
from project.metrics import r2_weighted

TEST_SIZE = 200
GAP = 0


class FullPipeline:
    def __init__(
        self,
        model: BaseEstimator,
        preprocessor=None,
        run_name: str = "",
        name: str = "",
        load_model: bool = False,
        features: list | None = None,
        save_to_disc: bool = True,
        refit=True,
        change_lr=False,
        col_target=CFG.COL_TARGET,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.name = name
        self.load_model = load_model
        self.features = features
        self.save_to_disc = save_to_disc
        self.refit = refit
        self.change_lr = change_lr
        self.col_target = col_target

        self.responders = [i for i in CFG.COL_RESPONDERS if i != self.col_target]
        self.set_run_name(run_name)
        self.path = os.path.join(CFG.PATH_MODELS, f"{self.run_name}")

    def set_run_name(self, run_name: str) -> None:
        self.run_name = run_name
        self.path = os.path.join(CFG.PATH_MODELS, f"{self.run_name}")
        if self.save_to_disc:
            utils.create_folder(self.path)

    def fit(
        self,
        df: pl.DataFrame | None = None,
        df_valid: pl.DataFrame | None = None,
        verbose: bool = False,
    ) -> None:
        if not self.load_model:
            self.model.features = self.features
            weights_train = df.select(CFG.COL_WEIGHT).to_series().to_numpy()
            dates_train = df.select(CFG.COL_DATE).to_series().to_numpy()
            times_train = df.select(CFG.COL_TIME).to_series().to_numpy()
            stocks_train = df.select(CFG.COL_ID).to_series().to_numpy()

            weights_valid = df_valid.select(CFG.COL_WEIGHT).to_series().to_numpy()
            dates_valid = df_valid.select(CFG.COL_DATE).to_series().to_numpy()
            times_valid = df_valid.select(CFG.COL_TIME).to_series().to_numpy()
            stocks_valid = df_valid.select(CFG.COL_ID).to_series().to_numpy()

            if self.preprocessor is not None:
                df = self.preprocessor.fit_transform(df)
                df_vaid = self.preprocessor.transform(df_valid)

            X_train = df.select(self.features).to_numpy()
            resp_train = df.select(self.responders).to_numpy()
            y_train = df.select(self.col_target).to_series().to_numpy()

            X_valid = df_valid.select(self.features).to_numpy()
            resp_valid = df_valid.select(self.responders).to_numpy()
            y_valid = df_valid.select(self.col_target).to_series().to_numpy()

            train_set = (
                X_train,
                resp_train,
                y_train,
                weights_train,
                stocks_train,
                dates_train,
                times_train,
            )

            val_set = (
                X_valid,
                resp_valid,
                y_valid,
                weights_valid,
                stocks_valid,
                dates_valid,
                times_valid,
            )

            del df, df_valid
            gc.collect()
            self.model.fit(train_set, val_set, verbose)
            if self.save_to_disc:
                self.save()
        else:
            self.load()

    def predict(
        self,
        df: pl.DataFrame,
        hidden: torch.Tensor | list | None = None,
        n_times: int | None = None,
    ) -> tuple[np.ndarray, torch.Tensor | list]:
        if n_times is None:
            n_times = len(df.select(CFG.COL_TIME).unique())
        if self.preprocessor is not None:
            df = self.preprocessor.transform(df)
        X = df.select(self.features).to_numpy()
        preds, hidden = self.model.predict(X, hidden=hidden, n_times=n_times)
        preds = np.clip(preds, -5, 5)
        return preds, hidden

    def update(self, df: pl.DataFrame) -> None:
        weights = df.select(CFG.COL_WEIGHT).to_series().to_numpy()
        n_times = len(df.select(CFG.COL_TIME).unique())
        if self.preprocessor is not None:
            df = self.preprocessor.transform(df, refit=True)
        X = df.select(self.features).to_numpy()
        y = df.select(self.col_target).to_series().to_numpy()
        self.model.update(X, y, weights, n_times)

    def load(self) -> None:
        if self.change_lr:
            lr_refit = self.model.lr_refit
        self.model = joblib.load(f"{self.path}/model_{self.name}.joblib")
        self.features = self.model.features
        if self.change_lr:
            self.model.lr_refit = lr_refit
        try:
            self.preprocessor = joblib.load(
                f"{self.path}/preprocessor_{self.name}.joblib"
            )
        except FileNotFoundError:
            self.preprocessor = None
            print("warnings:preprocessor not found")

    def save(self) -> None:
        joblib.dump(self.model, f"{self.path}/model_{self.name}.joblib")
        if self.preprocessor is not None:
            joblib.dump(
                self.preprocessor, f"{self.path}/preprocessor_{self.name}.joblib"
            )

    def get_params(self, deep: bool = True) -> dict:
        return {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "name": self.name,
            "load_model": self.load_model,
            "features": self.features,
            "save_to_disc": self.save_to_disc,
            "refit": self.refit,
            "change_lr": self.change_lr,
            "col_target": self.col_target,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.item():
            setattr(self, parameter, value)
        return self


class PipelineEnsemble:
    def __init__(
        self,
        models: list,
        weights: np.array = None,
        refit_models: list[bool] = None,
        col_target: str = CFG.COL_TARGET,
    ) -> None:
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(self.models))
        self.refit_models = (
            refit_models if refit_models is not None else [True] * len(models)
        )
        self.col_target = col_target
        self.refit = True

    def fit(
        self,
        df: pl.DataFrame,
        df_valid: pl.DataFrame,
        verbose: bool = False,
    ) -> None:
        self.weights = np.array(self.weights) / sum(self.weights)
        for model in self.models:
            model.fit(df, df_valid, verbose)

    def set_run_name(self, run_name: str) -> None:
        for model in self.models:
            model.set_run_name(run_name)

    def predict(self, df: pl.DataFrame, hidden_ls=None) -> np.ndarray:
        if hidden_ls is None:
            hidden_ls = [None] * len(self.models)
        preds = []
        for i, model in enumerate(self.models):
            preds_i, hidden_ls[i] = model.predict(df, hidden=hidden_ls[i])
            preds.append(preds_i)
        preds = np.average(preds, axis=0, weights=self.weights)
        return preds, hidden_ls

    def update(self, df: pl.DataFrame) -> None:
        for i, model in enumerate(self.models):
            if self.refit_models[i]:
                model.update(df)

    def load(self) -> None:
        for model in self.models:
            model.model.load()

    def save(self) -> None:
        for model in self.models:
            model.model.save()

    def get_params(self, deep: bool = True) -> dict:
        return {
            "models": self.models,
            "weights": self.weights,
            "refit_models": self.refit_models,
            "col_target": self.col_target,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PipelineCV:
    def __init__(
        self, model: FullPipeline, n_splits: int, train_size: int = False
    ) -> None:
        self.model = model
        self.n_splits = n_splits
        self.train_size = train_size
        self.models = []

    def fit(self, df: pl.DataFrame, verbose: bool = False) -> list:
        dates_unique = (
            df.select(pl.col(CFG.COL_DATE).unique().sort()).to_series().to_numpy()
        )
        test_size = (
            TEST_SIZE
            if len(dates_unique) > TEST_SIZE * (self.n_splits + 1)
            else len(dates_unique) // (self.n_splits + 1)
        )
        cv = TimeSeriesSplit(
            n_splits=self.n_splits, test_size=test_size, max_train_size=self.train_size
        )
        cv_split = cv.split(dates_unique)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv_split):
            if verbose:
                print("-" * 20 + f"Fold {fold}" + "-" * 20)
                print(
                    f"Train dates from {dates_unique[train_idx].min()}"
                    f"to {dates_unique[train_idx].max()}"
                )
                print(
                    f"Valid dates from {dates_unique[valid_idx].min()}"
                    f" to {dates_unique[valid_idx].max()}"
                )
            dates_train = dates_unique[train_idx]
            dates_valid = dates_unique[valid_idx]

            df_train = df.filter(pl.col(CFG.COL_DATE).is_in(dates_train))
            df_valid = df.filter(pl.col(CFG.COL_DATE).is_in(dates_valid))

            model_fold = clone(self.model)
            model_fold.set_run_name(f"fold{fold}")
            model_fold.fit(df_train, df_valid, verbose=verbose)

            self.models.append(model_fold)

            preds = []
            cnt_dates = 0
            model_save = copy.deepcopy(model_fold)
            for date_id in tqdm(dates_valid):

                df_valid_date = df_valid.filter(pl.col(CFG.COL_DATE) == date_id)

                if model_fold.refit & (cnt_dates > 0):
                    df_upd = df.filter(pl.col(CFG.COL_DATE) == date_id - 1)
                    if len(df_upd) > 0:
                        model_save.update(df_upd)

                preds_i, _ = model_save.predict(df_valid_date)
                preds += list(preds_i)
                cnt_dates += 1

            preds = np.array(preds)
            df_valid = df_valid.fill_null(0.0)
            y_true = (
                df_valid.select(pl.col(model_fold.col_target)).to_series().to_numpy()
            )
            weights = df_valid.select(pl.col(CFG.COL_WEIGHT)).to_series().to_numpy()
            score = r2_weighted(y_true, preds, weights)
            scores.append(score)

    def load(self) -> None:
        self.models = []
        for i in range(self.n_splits):
            model = clone(self.model)
            model.set_run_name(f"fold{i}")
            model.fit()
            self.models.append(model)
