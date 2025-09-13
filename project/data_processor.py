import os

import joblib
import polars as pl

from project import utils
from project.config import CFG
from project.transformers import PolarsTransformer


class DataProcessor:
    PATH = os.path.join(CFG.PATH_MODELS, "data_processors")

    COL_FEATURES_INIT = [f"feature_{i:02d}" for i in range(79)]

    COL_FEATURES_CORR = [
        "feature_06",
        "feature_04",
        "feature_07",
        "feature_36",
        "feature_60",
        "feature_45",
        "feature_56",
        "feature_05",
        "feature_51",
        "feature_19",
        "feature_66",
        "feature_59",
        "feature_54",
        "feature_70",
        "feature_71",
        "feature_72",
    ]
    COL_FEATURES_CAT = ["feature_09", "feature_10", "feature_11"]

    T = 1000

    def __init__(
        self,
        name: str,
        skip_days: int = None,
        transformer: PolarsTransformer | None = None,
    ):
        self.name = name
        self.skip_days = skip_days
        self.transformer = transformer
        self.features = list(self.COL_FEATURES_INIT)
        self.features += [
            f"{i}_diff_rolling_avg_{self.T}" for i in self.COL_FEATURES_CORR
        ]
        self.features += [f"{i}_rolling_std_{self.T}" for i in self.COL_FEATURES_CORR]
        self.features += [f"{i}_avg_per_date_time" for i in self.COL_FEATURES_CORR]
        self.features += ["feature_time_id"]
        self.features = [i for i in self.features if i not in self.COL_FEATURES_CAT]
        utils.create_folder(self.PATH)

    def get_train_data(self) -> pl.DataFrame:
        df = self._load_data().collect()
        df = df.with_columns(
            (
                pl.col("responder_8")
                + pl.col("responder_8").shift(-4).over(["symbol_id"])
            )
            .fill_null(0.0)
            .alias("responder_9"),
            (
                pl.col("responder_6")
                + pl.col("responder_6").shift(-20).over(["symbol_id"])
                + pl.col("responder_6").shift(-40).over(["symbol_id"])
            )
            .fill_null(0.0)
            .alias("responder_10"),
        )
        df = self._add_features(df)

        if self.transformer is not None:
            self.transformer.set_features(self.features)
            df = self.transformer.fit_transform(df)
        self._save()
        return df

    def process_test_data(
        self,
        df: pl.DataFrame,
        fast: bool = False,
        date_id: int = 0,
        time_id: int = 0,
        symbols: list = None,
    ) -> pl.DataFrame:
        df = self._add_features(
            df, fast=fast, date_id=date_id, time_id=time_id, symbols=symbols
        )
        if self.transformer is not None:
            df = self.transformer.transform(df, refit=True)
        return df

    def _save(self):
        joblib.dump(self, f"{self.PATH}/{self.name}.joblib")

    def load(self) -> pl.DataFrame:
        return joblib.load(f"{self.PATH}/{self.name}.joblib")

    def _load_data(self) -> pl.DataFrame:
        df = pl.scan_parquet(f"{CFG.PATH_DATA}/train.parquet")
        df = df.drop("partition_id")
        if self.skip_days is not None:
            df = df.filter(pl.col("date_id") >= self.skip_days)
        return df

    def _add_features(
        self,
        df: pl.DataFrame,
        fast: bool = False,
        date_id: int | None = None,
        time_id: int | None = None,
        symbols: list = None,
    ) -> pl.DataFrame:
        df = self._get_window_average_std(
            df,
            self.COL_FEATURES_CORR,
            n=self.T,
            fast=fast,
            date_id=date_id,
            time_id=time_id,
            symbols=symbols,
        )
        df = self._get_market_average(df, self.COL_FEATURES_CORR, fast=fast)
        df = df.with_columns(pl.col("time_id").alias("feature_time_id"))
        return df

    def _get_window_average_std(
        self,
        df: pl.DataFrame,
        cols: list,
        n: int = 1000,
        fast: bool = False,
        date_id: int | None = None,
        time_id: int | None = None,
        symbols: list = None,
    ) -> pl.DataFrame:
        if not fast:
            df = df.with_columns(
                [
                    pl.col(col)
                    .rolling_mean(window_size=n)
                    .over(["symbol_id"])
                    .alias(f"{col}_rolling_avg_{n}")
                    for col in cols
                ]
                + [
                    pl.col(col)
                    .rolling_std(window_size=n)
                    .over(["symbol_id"])
                    .alias(f"{col}_rolling_std_{n}")
                    for col in cols
                ]
            )
        else:
            df = (
                df.group_by("symbol_id")
                .agg(
                    [pl.col(col).mean().alias(f"{col}_rolling_avg_{n}") for col in cols]
                    + [
                        pl.col(col).std().alias(f"{col}_rolling_std_{n}")
                        for col in cols
                    ]
                    + [
                        pl.col(col).last().alias(col)
                        for col in self.COL_FEATURES_INIT
                        + ["row_id", "weight", "is_scored"]
                    ]
                )
                .filter(pl.col("symbol_id").is_in(symbols))
            )
            df = df.with_columns(
                [
                    pl.lit(date_id).cast(pl.Int16).alias("date_id"),
                    pl.lit(time_id).cast(pl.Int16).alias("time_id"),
                ]
            )
        df = df.with_columns(
            [
                (pl.col(col) - pl.col(f"{col}_rolling_avg_{n}")).alias(
                    f"{col}_diff_rolling_avg_{n}"
                )
                for col in cols
            ]
        )
        df = df.drop([f"{col}_rolling_avg_{n}" for col in cols])
        return df

    def _get_market_average(
        self,
        df: pl.DataFrame,
        cols: list,
        fast: bool = False,
    ) -> pl.DataFrame:
        if not fast:
            df = df.with_columns(
                [
                    pl.col(col)
                    .mean()
                    .over(["date_id", "time_id"])
                    .alias(f"{col}_avg_per_date_time")
                    for col in cols
                ]
            )
        else:
            df = df.with_columns(
                [pl.col(col).mean().alias(f"{col}_avg_per_date_time") for col in cols]
            )
        return df
