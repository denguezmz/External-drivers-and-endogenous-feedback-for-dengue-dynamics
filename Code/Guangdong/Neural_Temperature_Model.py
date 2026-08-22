"""
Temperature-only neural network dengue transmission model.

The model retains the data, 365-day warm-up period, host-vector dynamics,
weekly aggregation, loss, parameter bounds, and evaluation scheme of the
empirical temperature model. The only mechanistic replacement is the biting
rate: b_emp(T) = 0.0043*T + 0.0943 is replaced by
b_NN(T) = Softplus[NN((T - 25) / 10)]. The network is pretrained against the
empirical curve and then jointly trained with weak empirical, monotonicity,
and smoothness priors.

The vector infection factor remains linear for 12.4 <= T < 26.1 and equals
one otherwise. Initial immature-vector abundance (Sp0_log10=7.17) and initial
infected-vector prevalence (rateIp0=2.4933e-08) remain fixed.
"""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration
SEED = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.environ.get("DENGUE_DATA_DIR", str(SCRIPT_DIR / "data"))
)
LOCAL_PATH = DATA_DIR / "Local.xlsx"
IMPORT_PATH = DATA_DIR / "Input.xlsx"
TEMP_PATH = DATA_DIR / "Temp.xlsx"
RAIN_PATH = DATA_DIR / "Rain.xlsx"
HALF_MONTH_TEMP_PATH = DATA_DIR / "Expanded_Half_Month_Averages.xlsx"
SMOOTHED_RAIN_PATH = DATA_DIR / "Seven_Day_Smoothed_Rain.xlsx"
BORN_WEIGHT_PATH = DATA_DIR / "yuan_state_dict_born.pth"

OUTPUT_DIR = Path(
    os.environ.get("DENGUE_OUTPUT_DIR", str(SCRIPT_DIR / "outputs"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_WORKBOOK = OUTPUT_DIR / "temperature_only_neural_network_model_with_prior_results.xlsx"
CHECKPOINT_PATH = OUTPUT_DIR / "temperature_only_neural_network_model_with_prior_checkpoint.pth"
TRAINING_HISTORY_CSV = OUTPUT_DIR / "temperature_only_neural_network_with_prior_training_history.csv"
FIT_FIGURE = OUTPUT_DIR / "temperature_only_neural_network_with_prior_weekly_fit.png"
SCATTER_FIGURE = OUTPUT_DIR / "temperature_only_neural_network_with_prior_observed_vs_predicted.png"
BITE_FIGURE = OUTPUT_DIR / "temperature_only_neural_network_with_prior_bite_curve.png"
HISTORY_FIGURE = OUTPUT_DIR / "temperature_only_neural_network_with_prior_training_history.png"

FIT_START_DAY = 365
WEEK_SIZE = 7
EPOCHS = 1000
LOG_INTERVAL = 10
MAX_GRAD_NORM = 1.0

# Match the empirical model's learning rate for dynamic parameters.
LR_DYNAMICS = 1e-3
# Use a smaller learning rate for the neural network.
LR_BITE = 1e-5

# Temperature priors
# Pretrain on the empirical function to start from a plausible positive response.
BITE_PRETRAIN_EPOCHS = 800
BITE_PRETRAIN_LR = 1e-3
BITE_PRETRAIN_LOG_INTERVAL = 100

# Apply priors only over the 10-40 degree C range of the empirical function.
PRIOR_TEMP_MIN = 10.0
PRIOR_TEMP_MAX = 40.0
PRIOR_GRID_POINTS = 121
EMPIRICAL_SLOPE = 0.0043
MIN_ALLOWED_SLOPE = 0.0010
SHAPE_VALIDATION_MIN_SLOPE = 0.0

# Normalize the case loss so the prior weights remain interpretable.
# The empirical prior is soft and allows the network to deviate from the curve.
EMPIRICAL_PRIOR_WEIGHT = 0.20
MONOTONICITY_WEIGHT = 0.50
SMOOTHNESS_WEIGHT = 0.01

NETWORK_HIDDEN_1 = 128
NETWORK_HIDDEN_2 = 128


# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
print(f"Using device: {DEVICE}")


# Data loading and utilities
def read_excel_column(path: Path, column: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_excel(path)
    if column not in df.columns:
        raise KeyError(
            f"Column {column!r} is missing from {path}; "
            f"available columns: {list(df.columns)}"
        )
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        bad_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Column {column!r} in {path} contains {bad_count} missing or "
            "non-finite values."
        )
    return values


def weekly_sum_complete_torch(
    x: torch.Tensor,
    window_size: int = WEEK_SIZE,
) -> torch.Tensor:
    """Sum complete seven-day weeks and discard any trailing partial week."""
    x = x.reshape(-1)
    n_complete = x.numel() // window_size
    if n_complete == 0:
        return x.new_empty((0,))
    used = x[: n_complete * window_size]
    return used.reshape(n_complete, window_size).sum(dim=1)


def weekly_sum_complete_numpy(
    x: np.ndarray,
    window_size: int = WEEK_SIZE,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n_complete = len(x) // window_size
    if n_complete == 0:
        return np.empty((0,), dtype=np.float64)
    used = x[: n_complete * window_size]
    return used.reshape(n_complete, window_size).sum(axis=1)


def weekly_mean_complete_numpy(
    x: np.ndarray,
    window_size: int = WEEK_SIZE,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n_complete = len(x) // window_size
    if n_complete == 0:
        return np.empty((0,), dtype=np.float64)
    used = x[: n_complete * window_size]
    return used.reshape(n_complete, window_size).mean(axis=1)


def torch_fit_metrics(
    observed: torch.Tensor,
    predicted: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    observed = observed.reshape(-1)
    predicted = predicted.reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted series have different lengths: "
            f"{observed.numel()} vs {predicted.numel()}"
        )

    residual = observed - predicted
    mse = torch.mean(residual ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(residual))
    ss_res = torch.sum(residual ** 2)
    ss_tot = torch.sum((observed - observed.mean()) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def numpy_fit_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted series have different lengths: "
            f"{len(observed)} vs {len(predicted)}"
        )
    if len(observed) == 0:
        raise ValueError("No data are available for goodness-of-fit metrics.")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed or predicted values contain NaN or infinity.")

    residual = observed - predicted
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if np.std(observed) > 0 and np.std(predicted) > 0:
        pearson_r = float(np.corrcoef(observed, predicted)[0, 1])
        regression_slope, regression_intercept = np.polyfit(observed, predicted, 1)
        regression_slope = float(regression_slope)
        regression_intercept = float(regression_intercept)
    else:
        pearson_r = float("nan")
        regression_slope = float("nan")
        regression_intercept = float("nan")

    return {
        "n": int(len(observed)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "regression_slope": regression_slope,
        "regression_intercept": regression_intercept,
        "observed_total": float(observed.sum()),
        "predicted_total": float(predicted.sum()),
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted.mean()),
        "observed_peak": float(observed.max()),
        "predicted_peak": float(predicted.max()),
        "mean_residual_observed_minus_predicted": float(residual.mean()),
    }


# Frozen climate-driven vector birth-rate network
class BORN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return torch.sigmoid(self.layer3(x))


def load_inputs() -> Dict[str, object]:
    local = read_excel_column(LOCAL_PATH, "All2")
    imported = read_excel_column(IMPORT_PATH, "All2")
    temperature = read_excel_column(TEMP_PATH, "All")
    rainfall = read_excel_column(RAIN_PATH, "All")
    half_month_temperature = read_excel_column(HALF_MONTH_TEMP_PATH, "All")
    smoothed_rain = read_excel_column(SMOOTHED_RAIN_PATH, "All")

    lengths = {
        "local": len(local),
        "imported": len(imported),
        "temperature": len(temperature),
        "rainfall": len(rainfall),
        "half_month_temperature": len(half_month_temperature),
        "smoothed_rain": len(smoothed_rain),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input series have different lengths: {lengths}")

    n_days = len(local)
    if n_days <= FIT_START_DAY:
        raise ValueError(
            f"The {n_days}-day series is too short for a "
            f"{FIT_START_DAY}-day warm-up period."
        )

    temperature_tensor = torch.tensor(
        temperature, device=DEVICE, dtype=DTYPE
    ).view(-1, 1)
    rainfall_tensor = torch.tensor(
        rainfall, device=DEVICE, dtype=DTYPE
    ).view(-1, 1)

    born = BORN().to(DEVICE)
    state_dict = torch.load(BORN_WEIGHT_PATH, map_location=DEVICE)
    born.load_state_dict(state_dict)
    born.eval()
    for parameter in born.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        born_input = torch.cat((temperature_tensor, rainfall_tensor), dim=1)
        mosquito_birth = born(born_input) * 18.0 + 4.0091

    # Match the empirical model's temperature-dependent vector ecology.
    ax = 0.0135
    hax = 28116.4141
    hhx = 35378.2344
    thx = 301.6750
    mup = 0.9991
    mua = 0.6308
    tp = 22.0
    vp = 20.0
    ta = 30.0
    va = 12.1170
    r = 0.2845
    zt = 21.7535

    T_np = temperature.astype(np.float64)
    AT_np = half_month_temperature.astype(np.float64)

    mu = (
        ax
        * ((T_np + 273.15) / 298.15)
        * np.exp(
            hax / 1.987
            * (1.0 / 298.15 - 1.0 / (T_np + 273.15))
        )
        / (
            1.0
            + np.exp(
                hhx / 1.987
                * (1.0 / thx - 1.0 / (T_np + 273.15))
            )
        )
    )
    mymu = np.minimum(np.where(AT_np > zt, mu, mu * r), 1.0)

    dp1 = np.abs(1.0 - mup * np.exp(-((T_np - tp) ** 2) / (vp ** 2)))
    dp2 = np.abs(1.0 - mup * np.exp(-((zt - tp) ** 2) / (vp ** 2)))
    mydp = np.minimum(np.where(AT_np > zt, dp1, dp2), 1.0)

    myda = np.minimum(
        np.abs(1.0 - mua * np.exp(-((T_np - ta) ** 2) / (va ** 2))),
        1.0,
    )

    return {
        "n_days": n_days,
        "local": torch.tensor(local, device=DEVICE, dtype=DTYPE),
        "imported": torch.tensor(imported, device=DEVICE, dtype=DTYPE),
        "temperature": temperature_tensor,
        "rainfall": rainfall_tensor,
        "smoothed_rain": torch.tensor(
            smoothed_rain, device=DEVICE, dtype=DTYPE
        ).view(-1, 1),
        "mosquito_birth": mosquito_birth,
        "mymu": torch.tensor(mymu, device=DEVICE, dtype=DTYPE).view(-1, 1),
        "mydp": torch.tensor(mydp, device=DEVICE, dtype=DTYPE).view(-1, 1),
        "myda": torch.tensor(myda, device=DEVICE, dtype=DTYPE).view(-1, 1),
        "born_state_dict": copy.deepcopy(born.state_dict()),
    }


# Temperature-only neural network b_NN(T)
class BiteRateNetwork(nn.Module):
    """Map temperature directly to a nonnegative, unbounded biting rate."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, NETWORK_HIDDEN_1),
            nn.Tanh(),
            nn.Linear(NETWORK_HIDDEN_1, NETWORK_HIDDEN_2),
            nn.Tanh(),
            nn.Linear(NETWORK_HIDDEN_2, 1),
        )

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

        # Initialize the output near 0.20 without an empirical-model checkpoint.
        initial_bite = 0.20
        inverse_softplus = math.log(math.expm1(initial_bite))
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.net[-1].bias, inverse_softplus)

    def forward(self, temperature_c: torch.Tensor) -> torch.Tensor:
        original_shape = temperature_c.shape
        normalized_temperature = (
            temperature_c.reshape(-1, 1) - 25.0
        ) / 10.0
        raw_output = self.net(normalized_temperature)
        bite_rate = F.softplus(raw_output)
        return bite_rate.reshape(original_shape)


# Matched host-vector transmission model
class TemperatureOnlyNeuralDengueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "human_population",
            torch.tensor(113_460_000.0, dtype=DTYPE),
        )

        # Keep these initial values fixed, as in the empirical model.
        self.Sp0_log10 = nn.Parameter(
            torch.tensor(7.17, dtype=DTYPE),
            requires_grad=False,
        )
        self.rateIp0 = nn.Parameter(
            torch.tensor(2.4933e-08, dtype=DTYPE),
            requires_grad=False,
        )

        # Use the empirical model's initial values.
        self.gamma = nn.Parameter(torch.tensor(0.1012, dtype=DTYPE))
        self.out = nn.Parameter(torch.tensor(0.01, dtype=DTYPE))
        self.U = nn.Parameter(torch.tensor(0.9436, dtype=DTYPE))
        self.detah = nn.Parameter(torch.tensor(0.1037, dtype=DTYPE))

        self.bite_network = BiteRateNetwork()

    @staticmethod
    def empirical_bite_rate(temperature_c: torch.Tensor) -> torch.Tensor:
        """Return the reference curve for post-training plots only."""
        return torch.where(
            (temperature_c >= 10.0) & (temperature_c <= 40.0),
            0.0043 * temperature_c + 0.0943,
            torch.zeros_like(temperature_c),
        )

    @staticmethod
    def human_infection_factor(temperature_c: torch.Tensor) -> torch.Tensor:
        inside = (temperature_c >= 12.286) & (temperature_c <= 32.461)
        value = (
            0.001044
            * temperature_c
            * (temperature_c - 12.286)
            * torch.sqrt(torch.clamp(32.461 - temperature_c, min=0.0))
        )
        return torch.where(inside, value, torch.zeros_like(temperature_c))

    @staticmethod
    def mosquito_infection_factor(temperature_c: torch.Tensor) -> torch.Tensor:
        """Match the empirical model: linear on [12.4, 26.1), else one."""
        linear_part = -0.9037 + 0.0729 * temperature_c
        return torch.where(
            (temperature_c >= 12.4) & (temperature_c < 26.1),
            linear_part,
            torch.ones_like(temperature_c),
        )

    def learned_bite_rate(self, temperature_c: torch.Tensor) -> torch.Tensor:
        return self.bite_network(temperature_c)

    def clamp_parameters(self) -> None:
        """Apply the empirical model's dynamic-parameter bounds."""
        with torch.no_grad():
            self.gamma.clamp_(0.0714, 0.3333)
            self.U.clamp_(0.826, 0.9436)
            self.detah.clamp_(0.10, 0.25)
            self.out.clamp_(0.01, 0.03)

    def forward(
        self,
        mymu: torch.Tensor,
        mydp: torch.Tensor,
        myda: torch.Tensor,
        mosquito_birth: torch.Tensor,
        smoothed_rain: torch.Tensor,
        imported: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_days = temperature.shape[0]
        device = temperature.device
        dtype = temperature.dtype

        Sp0 = torch.pow(
            torch.tensor(10.0, device=device, dtype=dtype),
            self.Sp0_log10,
        )
        SpK = Sp0 * 5.0

        zero = torch.zeros((), device=device, dtype=dtype)
        initial_state = torch.stack(
            [
                self.human_population,                 # S
                zero,                                  # E
                zero,                                  # I symptomatic
                zero,                                  # A asymptomatic
                zero,                                  # Iin imported
                zero,                                  # R
                Sp0 * (1.0 - self.rateIp0),            # Sp
                Sp0 * self.rateIp0,                    # Ip
                zero,                                  # Sa
                zero,                                  # Ea
                zero,                                  # Ia
            ]
        )

        # Generate b_NN(T) for the full temperature series in one pass.
        bite_series = self.learned_bite_rate(temperature).reshape(-1)
        temperature_flat = temperature.reshape(-1)
        beta_h_series = (
            bite_series
            * self.human_infection_factor(temperature_flat)
        )
        beta_v_series = (
            bite_series
            * self.mosquito_infection_factor(temperature_flat)
        )
        detaa_series = 1.0 / (
            4.0 + torch.exp(5.15 - 0.123 * temperature_flat)
        )

        internal_state = initial_state
        states: List[torch.Tensor] = []
        diagnostics: List[torch.Tensor] = []

        for step in range(n_days):
            S, E, I, A, Iin, R, Sp, Ip, Sa, Ea, Ia = internal_state
            N = torch.clamp(S + E + I + A + R + Iin, min=1.0)

            birth_step = mosquito_birth[step].squeeze()
            mymu_step = mymu[step].squeeze()
            mydp_step = mydp[step].squeeze()
            myda_step = myda[step].squeeze()
            rain_step = smoothed_rain[step].squeeze()
            imported_step = imported[step].squeeze()
            beta_h = beta_h_series[step]
            beta_v = beta_v_series[step]
            detaa = detaa_series[step]
            bite = bite_series[step]

            # Vector dynamics match the empirical model.
            dSp = (
                birth_step * Sa
                + self.U * birth_step * (Ia + Ea)
                - mydp_step * Sp
                - mymu_step * Sp
            )
            dIp = (
                (1.0 - self.U) * birth_step * (Ia + Ea)
                - mydp_step * Ip
                - mymu_step * Ip
            )
            emergence = torch.abs(
                torch.exp(
                    -0.1
                    * (
                        1.0
                        + (mymu_step * Sp + mymu_step * Ip)
                        / (SpK * (1.0 + 0.0239 * rain_step))
                    )
                )
            )

            force_h_to_v = beta_v * Sa * (I + A + Iin) / N
            dSa = emergence * mymu_step * Sp - force_h_to_v - myda_step * Sa
            dEa = force_h_to_v - myda_step * Ea - detaa * Ea
            dIa = emergence * mymu_step * Ip + detaa * Ea - myda_step * Ia

            # Human dynamics match the empirical model.
            force_v_to_h = beta_h * Ia * S / N
            dS = -force_v_to_h
            dE = force_v_to_h - self.detah * E
            dI = 0.3125 * self.detah * E - self.gamma * I
            dA = 0.6875 * self.detah * E - self.gamma * A
            dIin = imported_step - (self.out + self.gamma) * Iin
            dR = (
                self.gamma * I
                + (self.out + self.gamma) * Iin
                + self.gamma * A
            )

            next_state = torch.stack(
                [
                    torch.clamp(S + dS, min=0.0),
                    torch.clamp(E + dE, min=0.0),
                    torch.clamp(I + dI, min=0.0),
                    torch.clamp(A + dA, min=0.0),
                    torch.clamp(Iin + dIin, min=0.0),
                    torch.clamp(R + dR, min=0.0),
                    torch.clamp(Sp + dSp, min=0.0),
                    torch.clamp(Ip + dIp, min=0.0),
                    torch.clamp(Sa + dSa, min=0.0),
                    torch.clamp(Ea + dEa, min=0.0),
                    torch.clamp(Ia + dIa, min=0.0),
                ]
            )

            # Incident symptomatic cases follow 0.3125 * detah * E.
            predicted_daily_cases = 0.3125 * self.detah * E

            states.append(next_state)
            diagnostics.append(
                torch.stack(
                    [
                        predicted_daily_cases,
                        beta_h,
                        beta_v,
                        bite,
                        temperature_flat[step],
                    ]
                )
            )
            internal_state = next_state

        return torch.stack(states), torch.stack(diagnostics)


# Losses and training
def temperature_prior_losses(
    model: TemperatureOnlyNeuralDengueModel,
) -> Dict[str, torch.Tensor]:
    """Calculate soft empirical-fit, monotonicity, and smoothness priors."""
    temperature_grid = torch.linspace(
        PRIOR_TEMP_MIN,
        PRIOR_TEMP_MAX,
        PRIOR_GRID_POINTS,
        device=DEVICE,
        dtype=DTYPE,
    )
    predicted_bite = model.learned_bite_rate(temperature_grid)
    empirical_bite = model.empirical_bite_rate(temperature_grid).detach()

    empirical_scale = empirical_bite.mean().clamp_min(1e-6)
    empirical_prior = torch.mean(
        ((predicted_bite - empirical_bite) / empirical_scale) ** 2
    )

    delta_temperature = temperature_grid[1] - temperature_grid[0]
    local_slopes = (predicted_bite[1:] - predicted_bite[:-1]) / delta_temperature
    slope_deficit = F.relu(MIN_ALLOWED_SLOPE - local_slopes)
    monotonicity = torch.mean(
        (slope_deficit / EMPIRICAL_SLOPE) ** 2
    )

    second_difference = (
        predicted_bite[2:]
        - 2.0 * predicted_bite[1:-1]
        + predicted_bite[:-2]
    )
    smoothness = torch.mean(
        (second_difference / empirical_scale) ** 2
    )

    return {
        "empirical_prior": empirical_prior,
        "monotonicity": monotonicity,
        "smoothness": smoothness,
        "minimum_slope": local_slopes.min(),
        "mean_slope": local_slopes.mean(),
    }


def pretrain_bite_network_to_empirical_prior(
    model: TemperatureOnlyNeuralDengueModel,
) -> pd.DataFrame:
    """Pretrain b_NN(T) to initialize the subsequent dynamic calibration."""
    optimizer = torch.optim.Adam(
        model.bite_network.parameters(),
        lr=BITE_PRETRAIN_LR,
    )

    temperature_grid = torch.linspace(
        PRIOR_TEMP_MIN,
        PRIOR_TEMP_MAX,
        PRIOR_GRID_POINTS,
        device=DEVICE,
        dtype=DTYPE,
    )
    target_bite = model.empirical_bite_rate(temperature_grid).detach()
    target_scale = target_bite.mean().clamp_min(1e-6)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.bite_network.state_dict())

    for epoch in range(BITE_PRETRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        predicted_bite = model.learned_bite_rate(temperature_grid)
        fit_loss = torch.mean(
            ((predicted_bite - target_bite) / target_scale) ** 2
        )

        second_difference = (
            predicted_bite[2:]
            - 2.0 * predicted_bite[1:-1]
            + predicted_bite[:-2]
        )
        smoothness = torch.mean(
            (second_difference / target_scale) ** 2
        )
        total_loss = fit_loss + 0.01 * smoothness

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Bite pretraining epoch {epoch}: loss is NaN or infinite."
            )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.bite_network.parameters(),
            max_norm=MAX_GRAD_NORM,
        )
        optimizer.step()

        current_loss = float(total_loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.bite_network.state_dict())

        history.append({
            "phase": "bite_prior_pretraining",
            "epoch": epoch,
            "prior_fit_loss": float(fit_loss.detach().cpu()),
            "prior_smoothness": float(smoothness.detach().cpu()),
            "total_loss": current_loss,
        })

        if (
            epoch % BITE_PRETRAIN_LOG_INTERVAL == 0
            or epoch == BITE_PRETRAIN_EPOCHS - 1
        ):
            print(
                f"Bite prior pretraining [{epoch}/{BITE_PRETRAIN_EPOCHS}] | "
                f"loss={current_loss:.6f}"
            )

    model.bite_network.load_state_dict(best_state)
    print(f"Bite prior pretraining finished. Best loss={best_loss:.6f}.")
    return pd.DataFrame(history)


def extract_parameters(
    model: TemperatureOnlyNeuralDengueModel,
) -> Dict[str, float]:
    return {
        "Sp0_log10_fixed": float(model.Sp0_log10.detach().cpu()),
        "Sp0_fixed": float(10.0 ** model.Sp0_log10.detach().cpu().item()),
        "rateIp0_fixed": float(model.rateIp0.detach().cpu()),
        "gamma": float(model.gamma.detach().cpu()),
        "out": float(model.out.detach().cpu()),
        "U": float(model.U.detach().cpu()),
        "detah": float(model.detah.detach().cpu()),
    }


def train_model(
    model: TemperatureOnlyNeuralDengueModel,
    inputs: Dict[str, object],
) -> pd.DataFrame:
    model.train()

    observed_weekly = weekly_sum_complete_torch(
        inputs["local"][FIT_START_DAY:],
        WEEK_SIZE,
    ).detach()

    # Prior pretraining updates only the biting-rate network.
    pretraining_history = pretrain_bite_network_to_empirical_prior(model)

    # Normalize case MSE by observed weekly variance; retain raw MSE for evaluation.
    case_loss_scale = torch.var(
        observed_weekly,
        unbiased=False,
    ).detach().clamp_min(1.0)

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.bite_network.parameters(),
                "lr": LR_BITE,
            },
            {
                "params": [model.gamma, model.out, model.detah, model.U],
                "lr": LR_DYNAMICS,
            },
        ]
    )

    history: List[Dict[str, float]] = pretraining_history.to_dict(orient="records")
    best_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        _, diagnostics = model(
            inputs["mymu"],
            inputs["mydp"],
            inputs["myda"],
            inputs["mosquito_birth"],
            inputs["smoothed_rain"],
            inputs["imported"],
            inputs["temperature"],
        )

        predicted_weekly = weekly_sum_complete_torch(
            diagnostics[FIT_START_DAY:, 0],
            WEEK_SIZE,
        )

        metrics = torch_fit_metrics(observed_weekly, predicted_weekly)
        case_loss = metrics["mse"]
        normalized_case_loss = case_loss / case_loss_scale

        prior_losses = temperature_prior_losses(model)
        empirical_prior = prior_losses["empirical_prior"]
        monotonicity = prior_losses["monotonicity"]
        smoothness = prior_losses["smoothness"]

        total_loss = (
            normalized_case_loss
            + EMPIRICAL_PRIOR_WEIGHT * empirical_prior
            + MONOTONICITY_WEIGHT * monotonicity
            + SMOOTHNESS_WEIGHT * smoothness
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Epoch {epoch}: loss is NaN or infinite."
            )

        # Record the current state before optimizer.step().
        current_loss = float(case_loss.detach().cpu())
        current_normalized_case_loss = float(normalized_case_loss.detach().cpu())
        current_total_loss = float(total_loss.detach().cpu())
        current_r2 = float(metrics["r2"].detach().cpu())
        current_rmse = float(metrics["rmse"].detach().cpu())
        current_mae = float(metrics["mae"].detach().cpu())

        shape_valid = (
            float(prior_losses["minimum_slope"].detach().cpu())
            >= SHAPE_VALIDATION_MIN_SLOPE
        )
        if shape_valid and current_total_loss < best_loss:
            best_loss = current_total_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        total_loss.backward()
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=MAX_GRAD_NORM,
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Epoch {epoch}: gradient norm is NaN or infinite."
            )

        optimizer.step()
        model.clamp_parameters()

        parameter_values = extract_parameters(model)
        history.append(
            {
                "phase": "joint_dynamics_training",
                "epoch": epoch,
                "case_mse": current_loss,
                "normalized_case_loss": current_normalized_case_loss,
                "empirical_prior_loss": float(empirical_prior.detach().cpu()),
                "monotonicity_loss": float(monotonicity.detach().cpu()),
                "smoothness": float(smoothness.detach().cpu()),
                "minimum_bite_slope": float(prior_losses["minimum_slope"].detach().cpu()),
                "mean_bite_slope": float(prior_losses["mean_slope"].detach().cpu()),
                "shape_valid": shape_valid,
                "total_loss": current_total_loss,
                "weekly_r2": current_r2,
                "weekly_rmse": current_rmse,
                "weekly_mae": current_mae,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr_bite": float(optimizer.param_groups[0]["lr"]),
                "lr_dynamics": float(optimizer.param_groups[1]["lr"]),
                **parameter_values,
            }
        )

        if epoch % LOG_INTERVAL == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch [{epoch}/{EPOCHS}] | "
                f"MSE={current_loss:.6f} | "
                f"R2={current_r2:.4f} | "
                f"RMSE={current_rmse:.4f} | "
                f"MAE={current_mae:.4f} | "
                f"Prior={empirical_prior.item():.5f} | "
                f"Mono={monotonicity.item():.5f} | "
                f"MinSlope={prior_losses['minimum_slope'].item():.6f} | "
                f"gamma={model.gamma.item():.6f} | "
                f"U={model.U.item():.6f} | "
                f"detah={model.detah.item():.6f} | "
                f"out={model.out.item():.6f}"
            )

    # Restore the state with the lowest penalized objective during training.
    if best_epoch < 0:
        raise RuntimeError(
            "Training produced no candidate satisfying the nondecreasing "
            "constraint. Increase MONOTONICITY_WEIGHT or extend prior pretraining."
        )

    model.load_state_dict(best_state)
    model.clamp_parameters()
    final_prior = temperature_prior_losses(model)
    final_min_slope = float(final_prior["minimum_slope"].detach().cpu())
    if final_min_slope < SHAPE_VALIDATION_MIN_SLOPE:
        raise RuntimeError(
            "The final biting-rate curve still has a negative slope: "
            f"{final_min_slope:.8f}."
        )

    print(
        f"Training finished. Best epoch={best_epoch}, "
        f"best penalized objective={best_loss:.6f}."
    )

    return pd.DataFrame(history)


# Post-training evaluation and outputs
def evaluate_model(
    model: TemperatureOnlyNeuralDengueModel,
    inputs: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        states, diagnostics = model(
            inputs["mymu"],
            inputs["mydp"],
            inputs["myda"],
            inputs["mosquito_birth"],
            inputs["smoothed_rain"],
            inputs["imported"],
            inputs["temperature"],
        )

    states_np = states.detach().cpu().numpy()
    diagnostics_np = diagnostics.detach().cpu().numpy()
    local_np = inputs["local"].detach().cpu().numpy().reshape(-1)
    n_days = len(local_np)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")

    daily_df = pd.DataFrame(
        {
            "date": dates,
            "is_fit_period": np.arange(n_days) >= FIT_START_DAY,
            "S": states_np[:, 0],
            "E": states_np[:, 1],
            "I": states_np[:, 2],
            "A": states_np[:, 3],
            "Iin": states_np[:, 4],
            "R": states_np[:, 5],
            "Sp": states_np[:, 6],
            "Ip": states_np[:, 7],
            "Sa": states_np[:, 8],
            "Ea": states_np[:, 9],
            "Ia": states_np[:, 10],
            "predicted_daily_cases": diagnostics_np[:, 0],
            "observed_daily_cases": local_np,
            "betah": diagnostics_np[:, 1],
            "betav": diagnostics_np[:, 2],
            "bite_rate": diagnostics_np[:, 3],
            "temperature_c": diagnostics_np[:, 4],
        }
    )
    daily_df["residual_observed_minus_predicted"] = (
        daily_df["observed_daily_cases"]
        - daily_df["predicted_daily_cases"]
    )

    fit_daily = daily_df.iloc[FIT_START_DAY:].reset_index(drop=True)
    observed_weekly = weekly_sum_complete_numpy(
        fit_daily["observed_daily_cases"].to_numpy(),
        WEEK_SIZE,
    )
    predicted_weekly = weekly_sum_complete_numpy(
        fit_daily["predicted_daily_cases"].to_numpy(),
        WEEK_SIZE,
    )
    mean_temperature = weekly_mean_complete_numpy(
        fit_daily["temperature_c"].to_numpy(),
        WEEK_SIZE,
    )
    mean_bite = weekly_mean_complete_numpy(
        fit_daily["bite_rate"].to_numpy(),
        WEEK_SIZE,
    )
    mean_betah = weekly_mean_complete_numpy(
        fit_daily["betah"].to_numpy(),
        WEEK_SIZE,
    )
    mean_betav = weekly_mean_complete_numpy(
        fit_daily["betav"].to_numpy(),
        WEEK_SIZE,
    )

    n_weeks = len(observed_weekly)
    week_starts = pd.date_range(
        dates[FIT_START_DAY], periods=n_weeks, freq="7D"
    )

    weekly_df = pd.DataFrame(
        {
            "week_index": np.arange(n_weeks),
            "week_start": week_starts,
            "observed_weekly_cases": observed_weekly,
            "predicted_weekly_cases": predicted_weekly,
            "mean_temperature_c": mean_temperature,
            "mean_bite_rate": mean_bite,
            "mean_betah": mean_betah,
            "mean_betav": mean_betav,
        }
    )
    weekly_df["residual_observed_minus_predicted"] = (
        weekly_df["observed_weekly_cases"]
        - weekly_df["predicted_weekly_cases"]
    )
    weekly_df["absolute_error"] = np.abs(
        weekly_df["residual_observed_minus_predicted"]
    )
    weekly_df["squared_error"] = (
        weekly_df["residual_observed_minus_predicted"] ** 2
    )

    daily_metrics = {
        "scale": "daily_fit_period",
        **numpy_fit_metrics(
            fit_daily["observed_daily_cases"].to_numpy(),
            fit_daily["predicted_daily_cases"].to_numpy(),
        ),
    }
    weekly_metrics = {
        "scale": "weekly_fit_period_complete_weeks",
        **numpy_fit_metrics(observed_weekly, predicted_weekly),
    }
    metrics_df = pd.DataFrame([daily_metrics, weekly_metrics])

    temperature_grid = torch.linspace(
        0.0, 40.0, 401, device=DEVICE, dtype=DTYPE
    )
    with torch.no_grad():
        learned_bite = model.learned_bite_rate(temperature_grid)
        empirical_bite = model.empirical_bite_rate(temperature_grid)

    temperature_np = temperature_grid.detach().cpu().numpy()
    empirical_np = empirical_bite.detach().cpu().numpy()
    learned_np = learned_bite.detach().cpu().numpy()
    local_slope_np = np.gradient(learned_np, temperature_np)

    bite_curve_df = pd.DataFrame(
        {
            "temperature_c": temperature_np,
            "empirical_reference_bite_rate": empirical_np,
            "neural_temperature_only_bite_rate": learned_np,
            "neural_minus_empirical": learned_np - empirical_np,
            "neural_local_slope_per_c": local_slope_np,
        }
    )

    return daily_df, weekly_df, metrics_df, bite_curve_df


def save_outputs(
    model: TemperatureOnlyNeuralDengueModel,
    inputs: Dict[str, object],
    history_df: pd.DataFrame,
) -> None:
    daily_df, weekly_df, metrics_df, bite_curve_df = evaluate_model(model, inputs)

    parameters = extract_parameters(model)
    parameters_df = pd.DataFrame(
        {
            "parameter": list(parameters.keys()),
            "value": list(parameters.values()),
        }
    )

    configuration_df = pd.DataFrame(
        {
            "item": [
                "seed",
                "device",
                "fit_start_day",
                "week_size",
                "epochs",
                "lr_bite",
                "lr_dynamics",
                "max_grad_norm",
                "bite_pretrain_epochs",
                "bite_pretrain_lr",
                "prior_temperature_range",
                "empirical_prior_weight",
                "monotonicity_weight",
                "minimum_allowed_slope",
                "shape_validation_min_slope",
                "smoothness_weight",
                "model_selection_objective",
                "network_architecture",
                "temperature_normalization",
                "output_constraint",
                "case_definition",
                "weekly_loss",
                "strict_bv_definition",
            ],
            "value": [
                SEED,
                str(DEVICE),
                FIT_START_DAY,
                WEEK_SIZE,
                EPOCHS,
                LR_BITE,
                LR_DYNAMICS,
                MAX_GRAD_NORM,
                BITE_PRETRAIN_EPOCHS,
                BITE_PRETRAIN_LR,
                f"{PRIOR_TEMP_MIN}-{PRIOR_TEMP_MAX} C",
                EMPIRICAL_PRIOR_WEIGHT,
                MONOTONICITY_WEIGHT,
                MIN_ALLOWED_SLOPE,
                SHAPE_VALIDATION_MIN_SLOPE,
                SMOOTHNESS_WEIGHT,
                "normalized weekly case MSE + weak empirical prior + monotonicity prior + smoothness prior",
                f"1-{NETWORK_HIDDEN_1}-{NETWORK_HIDDEN_2}-1; Tanh hidden layers",
                "(T - 25) / 10",
                "Softplus; nonnegative without fixed upper bound",
                "0.3125 * detah * E",
                "raw metric: unweighted weekly MSE; training objective uses variance-normalized weekly MSE plus temperature priors",
                "linear for 12.4<=T<26.1, otherwise 1",
            ],
        }
    )

    with pd.ExcelWriter(RESULT_WORKBOOK, engine="openpyxl") as writer:
        daily_df.to_excel(writer, sheet_name="DailyResults", index=False)
        weekly_df.to_excel(writer, sheet_name="WeeklyFit", index=False)
        metrics_df.to_excel(writer, sheet_name="FitMetrics", index=False)
        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
        bite_curve_df.to_excel(writer, sheet_name="BiteCurve", index=False)
        history_df.to_excel(writer, sheet_name="TrainingHistory", index=False)
        configuration_df.to_excel(writer, sheet_name="Configuration", index=False)

    history_df.to_csv(
        TRAINING_HISTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    weekly_metrics = metrics_df.loc[
        metrics_df["scale"] == "weekly_fit_period_complete_weeks"
    ].iloc[0]

    checkpoint = {
        "model_name": "temperature_only_neural_network_with_temperature_prior",
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "born_state_dict": inputs["born_state_dict"],
        "parameters": parameters,
        "weekly_metrics": weekly_metrics.to_dict(),
        "configuration": configuration_df.set_index("item")["value"].to_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    # Weekly case fit
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(
        weekly_df["week_start"],
        weekly_df["observed_weekly_cases"],
        s=18,
        label="Observed weekly cases",
    )
    ax.plot(
        weekly_df["week_start"],
        weekly_df["predicted_weekly_cases"],
        linewidth=2.0,
        label="Temperature-only neural model",
    )
    ax.set_title(
        "Temperature-only neural model | "
        f"R²={weekly_metrics['r2']:.4f}, "
        f"RMSE={weekly_metrics['rmse']:.3f}, "
        f"MAE={weekly_metrics['mae']:.3f}"
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Cases per week")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Observed-versus-predicted scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        weekly_df["observed_weekly_cases"],
        weekly_df["predicted_weekly_cases"],
        s=24,
        alpha=0.75,
    )
    max_value = float(
        max(
            weekly_df["observed_weekly_cases"].max(),
            weekly_df["predicted_weekly_cases"].max(),
        )
    )
    ax.plot([0.0, max_value], [0.0, max_value], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed weekly cases")
    ax.set_ylabel("Predicted weekly cases")
    ax.set_title(f"Observed vs predicted | R²={weekly_metrics['r2']:.4f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Temperature-biting-rate curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        bite_curve_df["temperature_c"],
        bite_curve_df["empirical_reference_bite_rate"],
        linestyle="--",
        linewidth=1.5,
        label="Empirical b(T) reference",
    )
    ax.plot(
        bite_curve_df["temperature_c"],
        bite_curve_df["neural_temperature_only_bite_rate"],
        linewidth=2.0,
        label="Neural b(T)",
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Biting rate")
    ax.set_title("Temperature-dependent biting-rate functions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(BITE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot weekly R-squared only for the joint dynamic-training phase.
    joint_history = history_df.loc[
        history_df["phase"] == "joint_dynamics_training"
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(joint_history["epoch"], joint_history["weekly_r2"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weekly R²")
    ax.set_title("Training history of weekly R²")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HISTORY_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nTraining and evaluation complete.")
    print(f"Results workbook: {RESULT_WORKBOOK}")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    print(f"Training log: {TRAINING_HISTORY_CSV}")
    print(f"Weekly fit plot: {FIT_FIGURE}")
    print(f"Scatter plot: {SCATTER_FIGURE}")
    print(f"Biting-rate curve: {BITE_FIGURE}")
    print(
        "Final weekly fit: "
        f"R²={weekly_metrics['r2']:.4f}, "
        f"MSE={weekly_metrics['mse']:.4f}, "
        f"RMSE={weekly_metrics['rmse']:.4f}, "
        f"MAE={weekly_metrics['mae']:.4f}, "
        f"Pearson r={weekly_metrics['pearson_r']:.4f}"
    )
    prior_summary = temperature_prior_losses(model)
    print(
        "Final temperature response: "
        f"minimum slope={prior_summary['minimum_slope'].item():.6f}, "
        f"mean slope={prior_summary['mean_slope'].item():.6f}"
    )


# Entry point
def main() -> None:
    inputs = load_inputs()
    model = TemperatureOnlyNeuralDengueModel().to(DEVICE)

    print("Starting independent training of the temperature-only neural model.")
    print(
        "Equations, state variables, case definition, warm-up, loss, and "
        "parameter bounds match the empirical model."
    )
    print("Mechanistic replacement: empirical b(T) -> neural b_NN(T).")
    print("Training uses weak empirical, monotonicity, and smoothness priors.")
    print(
        "Complete fitting weeks: "
        f"{(int(inputs['n_days']) - FIT_START_DAY) // WEEK_SIZE}"
    )

    history_df = train_model(model, inputs)
    save_outputs(model, inputs, history_df)


if __name__ == "__main__":
    main()
