import polars as pl


class PolarsTransformer:
    def __init__(
        self,
        features: list = None,
        fillnull: bool = True,
        scale: bool = True,
        clip_time: bool = True,
    ) -> None:
        self.features = features
        self.fillnull = fillnull
        self.scale = scale
        self.clip_time = clip_time
        self.statistics_mean_std = None
        self.statistics_min_max = None

    def set_features(self, features: list) -> None:
        self.features = features

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.scale:
            self.statistics_mean_std = {
                column: {"mean": df[column].mean(), "std": df[column].std()}
                for column in self.features
            }
        if self.clip_time:
            self.statistics_min_max = {
                column: {"min": df[column].min(), "max": df[column].max()}
                for column in ["feature_time_id"]
            }
        if self.fillnull:
            df = df.with_columns([pl.col(col).fill_null(0.0) for col in self.features])
        if self.scale:
            df = df.with_columns(
                [
                    (
                        (pl.col(col) - self.statistics_mean_std[col]["mean"])
                        / self.statistics_mean_std[col]["std"]
                    )
                    for col in self.features
                ]
            )
        return df

    def transform(self, df: pl.DataFrame, refit: bool = False) -> pl.DataFrame:
        if refit:
            if self.clip_time:
                self.statistics_min_max.update(
                    {
                        col: {
                            "min": (
                                self.statistics_min_max[col]["min"]
                                if df[col].min() is None
                                else min(
                                    df[col].min(), self.statistics_min_max[col]["min"]
                                )
                            ),
                            "max": (
                                self.statistics_min_max[col]["max"]
                                if df[col].max() is None
                                else max(
                                    df[col].max(), self.statistics_min_max[col]["max"]
                                )
                            ),
                        }
                        for col in ["feature_time_id"]
                    }
                )

        if self.clip_time:
            df = df.with_columns(
                [
                    pl.col(col).clip(
                        self.statistics_min_max[col]["min"],
                        self.statistics_min_max[col]["max"],
                    )
                    for col in ["feature_time_id"]
                ]
            )

        if self.fillnull:
            df = df.with_columns(
                [pl.col(column).fill_null(0.0) for column in self.features]
            )
        if self.scale:
            df = df.with_columns(
                [
                    (
                        (pl.col(column) - self.statistics_mean_std[column]["mean"])
                        / self.statistics_mean_std[column]["std"]
                    )
                    for column in self.features
                ]
            )
        return df
