import copy

import numpy as np
import torch
import torch.nn as nn
from project.metrics import WeightedR2Loss, r2_weighted_torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def flatten_collate_fn(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, resp, y, weights = zip(*batch)
    X = torch.cat(X, dim=0)
    resp = torch.cat(resp, dim=0)
    y = torch.cat(y, dim=0)
    weights = torch.cat(weights, dim=0)
    return X, resp, y, weights


class CustomTensorDataset(Dataset):
    T = 968

    def __init__(
        self,
        X: np.array,
        resp: np.array,
        y: np.array,
        weights: np.array,
        symbols: np.array,
        dates: np.array,
        times: np.array,
        on_batch: bool = True,
    ):

        self.on_batch = on_batch
        self.num_features = X.shape[1]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.resp = torch.tensor(resp, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.symbols = torch.tensor(symbols, dtype=torch.int64)
        self.dates = torch.tensor(dates, dtype=torch.int64)
        self.times = torch.tensor(times, dtype=torch.int64)

        self.X = torch.nan_to_num(self.X, 0)
        self.K = X.shape[1]

        if not self.on_batch:
            T = self.T
            N, K = self.X.shape

            sorted_indices = torch.argsort(self.times, stable=True)
            sorted_indices = sorted_indices[
                torch.argsort(self.dates[sorted_indices], stable=True)
            ]
            sorted_indices = sorted_indices[
                torch.argsort(self.symbols[sorted_indices], stable=True)
            ]

            self.X = self.X[sorted_indices]
            self.resp = self.resp[sorted_indices]
            self.dates = self.dates[sorted_indices]
            self.y = self.y[sorted_indices]
            self.weights = self.weights[sorted_indices]
            self.symbols = self.symbols[sorted_indices]

            self.X = self.X.view(N // T, T, K)
            self.resp = self.resp.view(N // T, T, self.resp.shape[-1])
            self.dates = self.dates.view(N // T, T)[:, 0].squeeze()
            self.y = self.y.view(N // T, T)
            self.weights = self.weights.view(N // T, T)
            self.symbols = self.symbols.view(N // T, T)

        self.datetime_ids = self.dates
        self.unique_datetimes, self.inverse_indices, self.counts = torch.unique(
            self.datetime_ids, return_inverse=True, return_counts=True
        )

        self.sorted_indices = torch.argsort(self.inverse_indices)
        self.group_end_indices = torch.cumsum(self.counts, dim=0)
        self.group_start_indices = torch.cat(
            (torch.tensor([0]), self.group_end_indices[:-1])
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        start = self.group_start_indices[index]
        end = self.group_end_indices[index]
        index = self.sorted_indices[start:end]

        X = self.X[index]
        resp = self.resp[index]
        y = self.y[index]
        weights = self.weights[index]

        if self.on_batch:
            T = max(self.times[index]) + 1
            X = X.reshape(T, -1, self.K).swapaxes(0, 1)
            resp = resp.reshape(T, -1, resp.shape[1]).swapaxes(0, 1)
            y = y.reshape(T, -1).swapaxes(0, 1)
            weights = weights.reshape(T, -1).swapaxes(0, 1)

        return X, resp, y, weights

    def __len__(self) -> int:
        return len(self.unique_datetimes)


class ModelRBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        dropout_rates: list,
        hidden_sizes_linear: list,
        dropout_rates_linear: list,
        model_type: str,
    ) -> None:
        super(ModelRBase, self).__init__()
        self.num_layers = len(hidden_sizes)

        self.gru_layers = nn.ModuleList()
        self.dropout_rates = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            if model_type == "gru":
                layer = nn.GRU(
                    input_dim, hidden_sizes[i], num_layers=1, batch_first=True
                )
            elif model_type == "lstm":
                layer = nn.LSTM(
                    input_dim, hidden_sizes[i], num_layers=1, batch_first=True
                )
            else:
                raise ValueError("Unknown model type")
            self.gru_layers.append(layer)
            self.dropout_rates.append(nn.Dropout(dropout_rates[i]))

        if self.num_layers == 0:
            n_input_linear = input_size
        else:
            n_input_linear = hidden_sizes[-1]

        fc_layers = []
        if hidden_sizes_linear:
            for i in range(len(hidden_sizes_linear)):
                in_features = n_input_linear if i == 0 else hidden_sizes_linear[i - 1]
                fc_layers.append(nn.Linear(in_features, hidden_sizes_linear[i]))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rates_linear[i]))
            fc_layers.append(nn.Linear(hidden_sizes_linear[-1], 1))
        else:
            fc_layers.append(nn.Linear(n_input_linear, 1))

        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self, x: torch.Tensor, hidden: bool = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D, T, _ = x.shape

        if hidden is None:
            hidden = [None] * self.num_layers

        for i, gru in enumerate(self.gru_layers):
            x, h = gru(x, hidden[i])
            if hasattr(self, "dropout_rates"):
                x = self.dropout_rates[i](x)
            hidden[i] = h

        x = x.reshape(D * T, -1)
        x = self.fc(x)
        x = x.reshape(D, T)

        return x, hidden


class ModelR(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        dropout_rates: list,
        hidden_sizes_linear: list,
        dropout_rates_linear: list,
        model_type: str,
    ):
        super(ModelR, self).__init__()
        self.num_resp = 4

        self.grus = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(self.num_resp):
            self.grus.append(
                ModelRBase(
                    input_size,
                    hidden_sizes,
                    dropout_rates,
                    hidden_sizes_linear,
                    dropout_rates_linear,
                    model_type,
                )
            )
        self.out = nn.Sequential(
            nn.Linear(self.num_resp, 1),
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D, T, _ = x.shape
        if hidden is None:
            hidden = [None] * self.num_resp

        out = []
        for i in range(len(self.grus)):
            z, h = self.grus[i](x, hidden[i])
            out.append(z)
            out[i] = out[i].reshape(D * T, -1)
            hidden[i] = h

        out_resp = torch.cat(out, dim=-1)
        y = self.out(out_resp)

        out_resp = out_resp.reshape(D, T, -1)
        y = y.reshape(D, T)
        return y, out_resp, hidden


class NN:
    def __init__(
        self,
        model_type: str | None = None,
        hidden_sizes: list | None = None,
        dropout_rates: list | None = None,
        hidden_sizes_linear: list | None = None,
        dropout_rates_linear: list | None = None,
        lr: float = 0.001,
        batch_size: int = 1,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        early_stopping: bool = True,
        lr_patience: int = 2,
        lr_factor: float = 0.5,
        lr_refit: float = 0.001,
        random_seed: int = 42,
    ):
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.hidden_sizes_linear = hidden_sizes_linear
        self.dropout_rates_linear = dropout_rates_linear
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = early_stopping
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_refit = lr_refit
        self.random_seed = random_seed

        self.criterion = WeightedR2Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.best_epoch = None
        self.features = None

    def fit(self, train_set: tuple, val_set: tuple, verbose: bool = False) -> None:
        torch.manual_seed(self.random_seed)
        train_dataset = CustomTensorDataset(*train_set, on_batch=False)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=flatten_collate_fn,
        )
        val_dataset = CustomTensorDataset(*val_set, on_batch=True)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=flatten_collate_fn,
        )
        self.model = ModelR(
            train_dataset.num_features,
            self.hidden_sizes,
            self.dropout_rates,
            self.hidden_sizes_linear,
            self.dropout_rates_linear,
            self.model_type,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        train_r2s, val_r2s = [], []
        if verbose:
            print(f"Device:{self.device}")
            print(
                f"{'Epoch':^5} | {'Train Loss':^10} | {'Val Loss':^8} "
                f"| {'Train R2':^9} | {'Val R2':^7} | {'LR':^7}"
            )
            print("-" * 60)

        min_val_r2 = -np.inf
        best_epoch = 0
        no_improvement = 0
        best_model = None
        for epoch in range(self.epochs):

            train_loss, train_r2 = self.train_one_epoch(train_dataloader, verbose)
            val_loss, val_r2 = self.validate_one_epoch(val_dataloader, verbose)
            lr_last = self.optimizer.param_groups[0]["lr"]

            train_r2s.append(train_r2)
            val_r2s.append(val_r2)

            if verbose:
                print(
                    f"{epoch+1:^5} | {train_loss:^10.4f} | {val_loss:^8.4f} "
                    f"{train_r2:^9.4f} | {val_r2:^7.4f} | {lr_last:^7.5f}"
                )
            if val_r2 > min_val_r2:
                min_val_r2 = val_r2
                best_model = copy.deepcopy(self.model.state_dict())
                no_improvement = 0
                best_epoch = epoch
            else:
                no_improvement += 1

            if self.early_stopping:
                if no_improvement >= self.early_stopping_patience + 1:
                    self.best_epoch = best_epoch + 1
                    if verbose:
                        print(
                            f"Early Stopping on epoch {best_epoch+1}"
                            f"Best score:{min_val_r2:4f}"
                        )
                    break

        if self.early_stopping:
            self.model.load_state_dict(best_model)

    def train_one_epoch(self, train_dataloader: DataLoader, verbose: bool) -> None:
        self.model.train()
        total_loss = 0.0
        y_total, weights_total, preds_total = [], [], []
        if verbose:
            itr = tqdm(train_dataloader)
        else:
            itr = train_dataloader
        for x_batch, resp_batch, y_batch, weights_batch in itr:
            x_batch, resp_batch, y_batch, weights_batch = (
                item.to(self.device)
                for item in [x_batch, resp_batch, y_batch, weights_batch]
            )
            self.optimizer.zero_grad()
            out_y, out_resp, _ = self.model(x_batch, None)
            loss1 = self.criterion(
                out_y.flatten(), y_batch.flatten(), weights_batch.flatten()
            )
            loss2 = self.criterion(
                out_resp[:, :, 0].flatten(),
                resp_batch[:, :, -1].flatten(),
                weights_batch.flatten(),
            )
            loss3 = self.criterion(
                out_resp[:, :, 1].flatten(),
                resp_batch[:, :, -2].flatten(),
                weights_batch.flatten(),
            )
            loss4 = self.criterion(
                out_resp[:, :, 2].flatten(),
                resp_batch[:, :, -3].flatten(),
                weights_batch.flatten(),
            )
            loss5 = self.criterion(
                out_resp[:, :, 3].flatten(),
                resp_batch[:, :, -4].flatten(),
                weights_batch.flatten(),
            )
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

            y_total.append(y_batch.flatten())
            weights_total.append(weights_batch.flatten())
            preds_total.append(out_y.detach().flatten())

        y_total = torch.cat(y_total).cpu()
        weights_total = torch.cat(weights_total).cpu()
        preds_total = torch.cat(preds_total).cpu()

        train_r2 = r2_weighted_torch(y_total, preds_total, weights_total).item()
        train_loss = total_loss / len(train_dataloader)

        return train_r2, train_loss

    def validate_one_epoch(self, val_dataloader: DataLoader, verbose: False) -> None:

        model = copy.deepcopy(self.model)

        losses, all_y, all_weights, all_preds = [], [], [], []

        if verbose:
            itr = tqdm(val_dataloader)
        else:
            itr = val_dataloader
        for x_batch, resp_batch, y_batch, weights_batch in itr:

            x_batch, resp_batch, y_batch, weights_batch = (
                item.to(self.device)
                for item in [x_batch, resp_batch, y_batch, weights_batch]
            )

            with torch.no_grad():
                model.eval()
                preds_batch, _, _ = model(x_batch, None)
                loss = self.criterion(
                    preds_batch.flatten(), y_batch.flatten(), weights_batch.flatten()
                )
                losses.append(loss.item())
                all_y.append(y_batch.flatten())
                all_weights.append(weights_batch.flatten())
                all_preds.append(preds_batch.flatten())

            if self.lr_refit > 0:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=self.lr_refit, weight_decay=0.01
                )
                optimizer.zero_grad()
                model.train()
                out_y, _, _ = model(x_batch, None)
                loss = self.criterion(
                    out_y.flatten(), y_batch.flatten(), weights_batch.flatten()
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        all_y = torch.cat(all_y)
        all_weights = torch.cat(all_weights)
        all_preds = torch.cat(all_preds)
        loss = np.mean(losses)
        r2 = r2_weighted_torch(all_y, all_preds, all_weights).item()

        return loss, r2

    def update(self, X: np.array, y: np.array, weights: np.array, n_times: int):
        if self.lr_refit == 0:
            return
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        N, K = X.shape
        X = X.view(n_times, N // n_times, K).swapaxes(0, 1).to(self.device)
        y = y.view(n_times, N // n_times).swapaxes(0, 1).to(self.device)
        weights = weights.view(n_times, N // n_times).swapaxes(0, 1).to(self.device)

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_refit, weight_decay=0.01
        )
        optimizer.zero_grad()
        out_y, _, _ = self.model(X, None)
        loss = self.criterion(out_y.flatten(), y.flatten(), weights.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

    def predict(
        self,
        X: np.array,
        n_times: int = None,
        hidden: torch.Tensor | list | None = None,
    ) -> tuple[np.array, torch.Tensor | list]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        N, K = X_tensor.shape
        X_tensor = (
            X_tensor.view(n_times, N // n_times, K).swapaxes(0, 1).to(self.device)
        )
        X_tensor = torch.nan_to_num(X_tensor, 0.0)
        self.model.eval()
        with torch.no_grad():
            preds, _, hidden = self.model(X_tensor, hidden)
            preds = preds.swapaxes(0, 1)
            preds = preds.reshape(-1).cpu().numpy()
        return preds, hidden

    def get_params(self, deep: bool = True):
        return {
            "model_type": self.model_type,
            "hidden_sizes": self.hidden_sizes,
            "dropout_rates": self.dropout_rates,
            "hidden_sizes_linear": self.hidden_sizes_linear,
            "dropout_rates_linear": self.dropout_rates_linear,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping": self.early_stopping,
            "lr_patience": self.lr_patience,
            "lr_factor": self.lr_factor,
            "lr_refit": self.lr_refit,
            "random_seed": self.random_seed,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
