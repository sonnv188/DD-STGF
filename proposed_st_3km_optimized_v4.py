from __future__ import annotations

import json
import math
import random
import gc
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
===========================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    DATA_PATH: str = (
        "D:/Sonnv/crimeprediction/data/28106939/"
        "crime_processed/daily_grid_counts"
    )

    OUTPUT_DIR: str = (
        "D:/Sonnv/crimeprediction/outputs/proposed_st_3km_optimized"
    )

 
    CACHE_DIR: str = (
        "D:/Sonnv/crimeprediction/cache/proposed_st_3km"
    )


    REBUILD_TEMPORAL_CACHE: bool = False

    DATE_COL: str = "event_date"
    GRID_ID_COL: str = "grid_id"
    CRIME_TYPE_COL: str = "crime_type_standardized"
    COUNT_COL: str = "crime_count"

    GRID_X_COL: str = "grid_center_x_m"
    GRID_Y_COL: str = "grid_center_y_m"
    GRID_ROW_COL: str = "grid_row"
    GRID_COL_COL: str = "grid_col"

    # -------------------------------------------------------------------------
    # 10 fixed crime classes
    # -------------------------------------------------------------------------
    CRIME_TYPES: Tuple[str, ...] = (
        "VIOLENT",
        "SEXUAL",
        "ROBBERY_EXTORTION",
        "PUBLIC_SAFETY",
        "DRUG",
        "TRAFFIC",
        "PUBLIC_ORDER",
        "FRAUD",
        "THEFT",
        "OTHER_CRIMINAL",
    )

    # -------------------------------------------------------------------------
    # Temporal split
    # -------------------------------------------------------------------------
    TRAIN_END: str = "2016-12-31"
    VAL_END: str = "2017-12-31"
    TEST_END: str = "2018-12-31"

    # -------------------------------------------------------------------------
    # Temporal inputs
    # -------------------------------------------------------------------------
    # Oldest -> newest so the final TCN position is the most recent day.
    RECENT_LAGS: Tuple[int, ...] = (7, 6, 5, 4, 3, 2, 1)
    # Oldest -> newest weekly observations.
    WEEKLY_LAGS: Tuple[int, ...] = (28, 21, 14, 7)
    # These are TEMPORAL SCALES, not a chronological sequence.
    LONG_WINDOWS: Tuple[int, ...] = (14, 30, 60)

    # -------------------------------------------------------------------------
    # Spatial graph
    # -------------------------------------------------------------------------
    GRID_SIZE_M: float = 3000.0

    SPATIAL_RADIUS_M: float = 4300.0
    SPATIAL_SIGMA_M: float = 3000.0
    INCLUDE_SELF_LOOPS: bool = True

    NUM_GCN_LAYERS: int = 2

    # -------------------------------------------------------------------------
    # Subgraph mini-batching
    # -------------------------------------------------------------------------

    CORE_NODES_PER_BATCH: int = 4096

    HALO_HOPS: int = 2

    TRAIN_ACTIVE_NODES_ONLY: bool = False

    HIDDEN_DIM: int = 64
    CRIME_EMBED_DIM: int = 32
    CONTEXT_DIM: int = 32
    DROPOUT: float = 0.15

    EPOCHS: int = 20
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    EARLY_STOPPING_PATIENCE: int = 4

    NUM_WORKERS: int = 0

    VAL_EVERY: int = 2

    USE_AMP: bool = True

    CACHE_SPATIAL_CHUNKS_ON_GPU: bool = True

    SPATIALLY_COHERENT_CHUNKS: bool = True

    PROFILE_TIMING: bool = True

    USE_LOG1P_INPUT: bool = True

    LOSS_NAME: str = "poisson"


    HUBER_DELTA: float = 1.0

    USE_SPATIAL_GRAPH: bool = True
    USE_CRIME_GRAPH: bool = True

    USE_CONTEXT_FEATURES: bool = True
    USE_CONTEXT_ADAPTIVE_CRIME_GRAPH: bool = True
    USE_STATE_ADAPTIVE_CRIME_GRAPH: bool = True

    USE_RECENT_STREAM: bool = True
    USE_WEEKLY_STREAM: bool = True
    USE_LONGTERM_STREAM: bool = True

    USE_TEMPORAL_ATTENTION: bool = True
    USE_GATED_FUSION: bool = True

    USE_TOPK_CRIME_GRAPH: bool = True
    CRIME_GRAPH_TOPK: int = 3

    RETURN_EXPLANATIONS: bool = True

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    MAX_VAL_BATCHES: Optional[int] = None
    MAX_TEST_BATCHES: Optional[int] = None


    HOTSPOT_FRACTIONS: Tuple[float, ...] = (0.01, 0.05, 0.10)


    SAVE_PLOTS: bool = True


CFG = Config()

def ensure_dir(path: str) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def read_csv_or_folder(path: str) -> pd.DataFrame:

    p = Path(path)

    if p.is_file():
        return pd.read_csv(p)

    if p.is_dir():
        files = sorted(
            f for f in p.glob("*.csv")
            if f.name.startswith("part-")
            or f.suffix.lower() == ".csv"
        )

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in: {path}"
            )

        frames = [
            pd.read_csv(f)
            for f in files
        ]

        return pd.concat(
            frames,
            ignore_index=True,
        )

    raise FileNotFoundError(path)


def inverse_target_transform(
    x: torch.Tensor,
    use_log1p: bool = False,
) -> torch.Tensor:

    return x.clamp_min(0.0)


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(
        torch.abs(pred - target)
    )


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(
            (pred - target) ** 2
        )
    )


def poisson_deviance_from_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean Poisson deviance for non-negative count predictions."""
    mu = np.clip(pred.astype(np.float64, copy=False), eps, None)
    y = np.clip(target.astype(np.float64, copy=False), 0.0, None)

    term = np.zeros_like(y, dtype=np.float64)
    nz = y > 0

    term[nz] = (
        y[nz] * np.log(y[nz] / mu[nz])
        - (y[nz] - mu[nz])
    )
    term[~nz] = mu[~nz]

    return float(2.0 * np.mean(term))


def r2_from_sufficient_statistics(
    sum_y: float,
    sum_y2: float,
    sum_sq_error: float,
    n: int,
) -> float:
    """R^2 without storing every prediction."""
    if n <= 1:
        return float("nan")

    sst = sum_y2 - (sum_y * sum_y) / n

    if sst <= 1e-12:
        return float("nan")

    return float(1.0 - sum_sq_error / sst)


def ndcg_at_k(
    predicted_scores: np.ndarray,
    observed_relevance: np.ndarray,
    k: int,
) -> float:
    """NDCG@K using observed total crime as graded relevance."""
    if k <= 0:
        return float("nan")

    k = min(k, len(predicted_scores))

    pred_order = np.argsort(
        -predicted_scores,
        kind="mergesort",
    )[:k]

    ideal_order = np.argsort(
        -observed_relevance,
        kind="mergesort",
    )[:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    dcg = float(
        np.sum(observed_relevance[pred_order] * discounts)
    )

    idcg = float(
        np.sum(observed_relevance[ideal_order] * discounts)
    )

    if idcg <= 0:
        return 0.0

    return dcg / idcg


def hotspot_metrics_for_one_day(
    predicted_total: np.ndarray,
    observed_total: np.ndarray,
    fractions: Sequence[float],
) -> Dict[str, float]:
    """
    Operational hotspot metrics:
      - HitRate@p
      - PAI@p
      - NDCG@p
    """
    result: Dict[str, float] = {}

    n = len(predicted_total)
    total_crime = float(observed_total.sum())

    for frac in fractions:
        k = max(1, int(math.ceil(n * frac)))

        top_idx = np.argpartition(
            -predicted_total,
            kth=min(k - 1, n - 1),
        )[:k]

        captured = float(observed_total[top_idx].sum())

        hit_rate = (
            captured / total_crime
            if total_crime > 0
            else 0.0
        )

        actual_area_frac = k / n

        pai = (
            hit_rate / actual_area_frac
            if actual_area_frac > 0
            else 0.0
        )

        pct = int(round(frac * 100))

        result[f"HitRate@{pct}%"] = float(hit_rate)
        result[f"PAI@{pct}%"] = float(pai)
        result[f"NDCG@{pct}%"] = ndcg_at_k(
            predicted_total,
            observed_total,
            k,
        )

    return result


def average_metric_dicts(
    rows: List[Dict[str, float]],
) -> Dict[str, float]:
    if not rows:
        return {}

    keys = sorted(rows[0].keys())

    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in keys
    }


# =============================================================================
# 4. LOAD AND STANDARDIZE SPARSE EVENT COUNTS
# =============================================================================

def load_and_standardize_data(
    cfg: Config,
) -> pd.DataFrame:

    df = read_csv_or_folder(
        cfg.DATA_PATH
    )

    required = {
        cfg.DATE_COL,
        cfg.GRID_ID_COL,
        cfg.CRIME_TYPE_COL,
        cfg.COUNT_COL,
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df[cfg.DATE_COL] = pd.to_datetime(
        df[cfg.DATE_COL],
        errors="coerce",
    )

    df[cfg.GRID_ID_COL] = (
        df[cfg.GRID_ID_COL]
        .astype(str)
    )

    df[cfg.COUNT_COL] = pd.to_numeric(
        df[cfg.COUNT_COL],
        errors="coerce",
    ).fillna(0.0)

    df = df[
        df[cfg.DATE_COL].notna()
        & df[cfg.CRIME_TYPE_COL].isin(
            cfg.CRIME_TYPES
        )
    ].copy()

    # Aggregate duplicate sparse rows.
    df = (
        df.groupby(
            [
                cfg.DATE_COL,
                cfg.GRID_ID_COL,
                cfg.CRIME_TYPE_COL,
            ],
            as_index=False,
        )[cfg.COUNT_COL]
        .sum()
    )

    return df


# =============================================================================
# 5. SPARSE TEMPORAL STORE
# =============================================================================

class SparseCrimeStore:

    CACHE_VERSION = 2

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
    ):
        self.cfg = cfg
        self.cache_dir = Path(cfg.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        min_date = pd.Timestamp(df[cfg.DATE_COL].min()).normalize()
        max_date = pd.Timestamp(df[cfg.DATE_COL].max()).normalize()

        self.dates = list(pd.date_range(min_date, max_date, freq="D"))
        self.grid_ids = sorted(df[cfg.GRID_ID_COL].astype(str).unique())
        self.crime_types = list(cfg.CRIME_TYPES)

        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.grid_to_idx = {g: i for i, g in enumerate(self.grid_ids)}
        self.crime_to_idx = {c: i for i, c in enumerate(self.crime_types)}

        self.T = len(self.dates)
        self.N = len(self.grid_ids)
        self.C = len(self.crime_types)

        print(
            "Temporal store dimensions: "
            f"T={self.T:,}, N={self.N:,}, C={self.C}"
        )

        grid_hash = hashlib.sha1(
            "\n".join(self.grid_ids).encode("utf-8")
        ).hexdigest()

        self.cache_meta = {
            "version": self.CACHE_VERSION,
            "T": self.T,
            "N": self.N,
            "C": self.C,
            "min_date": str(self.dates[0].date()),
            "max_date": str(self.dates[-1].date()),
            "grid_hash": grid_hash,
            "crime_types": self.crime_types,
            "long_windows": list(cfg.LONG_WINDOWS),
        }

        self.meta_path = self.cache_dir / "temporal_cache_meta.json"
        self.raw_path = self.cache_dir / "daily_counts_float32.dat"

        valid_cache = self._cache_is_valid()

        if cfg.REBUILD_TEMPORAL_CACHE or not valid_cache:
            print("Building 3-km temporal memmap cache...")
            self._build_raw_memmap(df)
            self._build_rolling_memmaps()

            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.cache_meta, f, ensure_ascii=False, indent=2)
        else:
            print("Reusing existing temporal memmap cache.")

        self.X = np.memmap(
            self.raw_path,
            mode="r",
            dtype=np.float32,
            shape=(self.T, self.N, self.C),
        )

        self.rolling: Dict[int, np.memmap] = {}
        for window in cfg.LONG_WINDOWS:
            path = self._rolling_path(window)
            self.rolling[int(window)] = np.memmap(
                path,
                mode="r",
                dtype=np.float32,
                shape=(self.T, self.N, self.C),
            )

        total_gib = (
            (1 + len(cfg.LONG_WINDOWS))
            * self.T * self.N * self.C * 4
            / (1024 ** 3)
        )
        print(
            f"Temporal memmaps ready (~{total_gib:.2f} GiB on disk; "
            "pages are loaded on demand)."
        )

    def _rolling_path(self, window: int) -> Path:
        return self.cache_dir / f"rolling_{int(window)}d_float32.dat"

    def _cache_is_valid(self) -> bool:
        if not self.meta_path.exists() or not self.raw_path.exists():
            return False

        try:
            old = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        if old != self.cache_meta:
            return False

        expected_raw = self.T * self.N * self.C * 4
        if self.raw_path.stat().st_size != expected_raw:
            return False

        for window in self.cfg.LONG_WINDOWS:
            p = self._rolling_path(window)
            if (not p.exists()) or p.stat().st_size != expected_raw:
                return False

        return True

    def _build_raw_memmap(self, df: pd.DataFrame) -> None:
        X = np.memmap(
            self.raw_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.T, self.N, self.C),
        )
        X[:] = 0.0

        dates_norm = pd.to_datetime(df[self.cfg.DATE_COL]).dt.normalize()
        t_idx = dates_norm.map(self.date_to_idx).to_numpy(dtype=np.int64)
        g_idx = (
            df[self.cfg.GRID_ID_COL]
            .astype(str)
            .map(self.grid_to_idx)
            .to_numpy(dtype=np.int64)
        )
        c_idx = (
            df[self.cfg.CRIME_TYPE_COL]
            .map(self.crime_to_idx)
            .to_numpy(dtype=np.int64)
        )
        vals = df[self.cfg.COUNT_COL].to_numpy(dtype=np.float32)

        X[t_idx, g_idx, c_idx] = vals
        X.flush()
        del X
        gc.collect()

    def _build_rolling_memmaps(self) -> None:
        X = np.memmap(
            self.raw_path,
            mode="r",
            dtype=np.float32,
            shape=(self.T, self.N, self.C),
        )

        for window in self.cfg.LONG_WINDOWS:
            window = int(window)
            print(f"Precomputing rolling mean: {window} days")

            out = np.memmap(
                self._rolling_path(window),
                mode="w+",
                dtype=np.float32,
                shape=(self.T, self.N, self.C),
            )

            running = np.zeros((self.N, self.C), dtype=np.float32)

            for t in range(self.T):
                if t > 0:
                    running += X[t - 1]

                old_t = t - window - 1
                if old_t >= 0:
                    running -= X[old_t]

                count = min(t, window)
                if count > 0:
                    out[t] = running / float(count)
                else:
                    out[t] = 0.0

                if (t + 1) % 365 == 0:
                    out.flush()

            out.flush()
            del out
            gc.collect()

        del X
        gc.collect()

    def get_day_nodes(
        self,
        t: int,
        node_ids: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            self.X[t, node_ids, :],
            dtype=np.float32,
        )

    def get_lag_sequence(
        self,
        target_t: int,
        lags: Sequence[int],
        node_ids: np.ndarray,
    ) -> np.ndarray:
        tids = np.asarray(
            [target_t - int(lag) for lag in lags],
            dtype=np.int64,
        )
        return np.asarray(
            self.X[np.ix_(tids, node_ids, np.arange(self.C))],
            dtype=np.float32,
        )

    def get_rolling_means(
        self,
        target_t: int,
        windows: Sequence[int],
        node_ids: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [
                np.asarray(
                    self.rolling[int(window)][target_t, node_ids, :],
                    dtype=np.float32,
                )
                for window in windows
            ],
            axis=0,
        )

    def active_training_nodes(
        self,
        last_train_t: int,
    ) -> np.ndarray:
        active = np.zeros(self.N, dtype=bool)

        for t in range(last_train_t + 1):
            active |= np.any(self.X[t] > 0, axis=1)

        return np.flatnonzero(active).astype(np.int64)


# =============================================================================
# 6. GRID METADATA
# =============================================================================

def extract_grid_metadata(
    raw_path: str,
    cfg: Config,
    grid_ids: List[str],
) -> pd.DataFrame:

    parsed = []

    for grid_id in grid_ids:
        parts = str(grid_id).split("_")

        try:
            size = float(parts[-3])
            col = int(parts[-2])
            row = int(parts[-1])

            if not np.isfinite(size) or size <= 0:
                raise ValueError

            parsed.append((str(grid_id), size, col, row))

        except (ValueError, TypeError, IndexError):
            parsed = []
            break

    if parsed:
        meta = pd.DataFrame(
            parsed,
            columns=[
                cfg.GRID_ID_COL,
                "_grid_size_from_id",
                cfg.GRID_COL_COL,
                cfg.GRID_ROW_COL,
            ],
        )

        sizes = meta["_grid_size_from_id"].to_numpy(dtype=np.float64)
        cols = meta[cfg.GRID_COL_COL].to_numpy(dtype=np.float64)
        rows = meta[cfg.GRID_ROW_COL].to_numpy(dtype=np.float64)

        meta[cfg.GRID_X_COL] = (cols + 0.5) * sizes
        meta[cfg.GRID_Y_COL] = (rows + 0.5) * sizes

        observed_sizes = sorted(set(meta["_grid_size_from_id"].astype(float)))
        print(f"Grid metadata parsed from grid_id; encoded sizes={observed_sizes}")

        if any(abs(s - cfg.GRID_SIZE_M) > 1e-6 for s in observed_sizes):
            print(
                "WARNING: GRID_SIZE_M differs from size encoded in grid_id. "
                "Spatial coordinates use the encoded grid size."
            )

        return meta

    # Fallback for non-standard grid IDs.
    print("grid_id parsing failed; falling back to source metadata read.")
    raw = read_csv_or_folder(raw_path)
    raw[cfg.GRID_ID_COL] = raw[cfg.GRID_ID_COL].astype(str)

    if cfg.GRID_X_COL in raw.columns and cfg.GRID_Y_COL in raw.columns:
        meta = (
            raw[[cfg.GRID_ID_COL, cfg.GRID_X_COL, cfg.GRID_Y_COL]]
            .drop_duplicates(cfg.GRID_ID_COL)
            .copy()
        )
    elif cfg.GRID_ROW_COL in raw.columns and cfg.GRID_COL_COL in raw.columns:
        meta = (
            raw[[cfg.GRID_ID_COL, cfg.GRID_ROW_COL, cfg.GRID_COL_COL]]
            .drop_duplicates(cfg.GRID_ID_COL)
            .copy()
        )

        meta[cfg.GRID_X_COL] = (
            (pd.to_numeric(meta[cfg.GRID_COL_COL], errors="coerce") + 0.5)
            * cfg.GRID_SIZE_M
        )
        meta[cfg.GRID_Y_COL] = (
            (pd.to_numeric(meta[cfg.GRID_ROW_COL], errors="coerce") + 0.5)
            * cfg.GRID_SIZE_M
        )
    else:
        raise ValueError(
            "Cannot obtain grid coordinates. Use grid_id format "
            "CN_AEA_<size>_<col>_<row> or retain metric coordinates."
        )

    meta = (
        meta.set_index(cfg.GRID_ID_COL)
        .reindex(grid_ids)
        .reset_index()
    )

    bad = meta[cfg.GRID_X_COL].isna() | meta[cfg.GRID_Y_COL].isna()
    if bad.any():
        raise ValueError(
            "Missing coordinates for grids: "
            f"{meta.loc[bad, cfg.GRID_ID_COL].head(10).tolist()}"
        )

    return meta


# =============================================================================
# 7. GLOBAL SPARSE SPATIAL GRAPH
# =============================================================================

class SparseSpatialGraph:

    def __init__(
        self,
        grid_meta: pd.DataFrame,
        cfg: Config,
    ):
        self.cfg = cfg

        xy = (
            grid_meta[
                [
                    cfg.GRID_X_COL,
                    cfg.GRID_Y_COL,
                ]
            ]
            .astype(np.float64)
            .to_numpy()
        )

        self.N = xy.shape[0]

        print(
            f"Building global sparse spatial graph "
            f"for {self.N:,} grids..."
        )

        tree = cKDTree(
            xy
        )

        pairs = tree.query_pairs(
            r=cfg.SPATIAL_RADIUS_M,
            output_type="ndarray",
        )

        if pairs.size == 0:
            pairs = np.empty(
                (0, 2),
                dtype=np.int64,
            )

        if len(pairs) > 0:
            diff = (
                xy[pairs[:, 0]]
                - xy[pairs[:, 1]]
            )

            dist = np.sqrt(
                np.sum(
                    diff * diff,
                    axis=1,
                )
            )

            w = np.exp(
                -dist
                / cfg.SPATIAL_SIGMA_M
            ).astype(np.float32)

            src = np.concatenate(
                [
                    pairs[:, 0],
                    pairs[:, 1],
                ]
            )

            dst = np.concatenate(
                [
                    pairs[:, 1],
                    pairs[:, 0],
                ]
            )

            val = np.concatenate(
                [
                    w,
                    w,
                ]
            )

        else:
            src = np.empty(
                0,
                dtype=np.int64,
            )
            dst = np.empty(
                0,
                dtype=np.int64,
            )
            val = np.empty(
                0,
                dtype=np.float32,
            )

        if cfg.INCLUDE_SELF_LOOPS:
            nodes = np.arange(
                self.N,
                dtype=np.int64,
            )

            src = np.concatenate(
                [src, nodes]
            )

            dst = np.concatenate(
                [dst, nodes]
            )

            val = np.concatenate(
                [
                    val,
                    np.ones(
                        self.N,
                        dtype=np.float32,
                    ),
                ]
            )

        self.A = sparse.coo_matrix(
            (
                val,
                (src, dst),
            ),
            shape=(
                self.N,
                self.N,
            ),
            dtype=np.float32,
        ).tocsr()

        self.A.sum_duplicates()

        degree = np.asarray(
            self.A.sum(axis=1)
        ).ravel().astype(np.float32)

        inv_sqrt = np.zeros_like(
            degree,
            dtype=np.float32,
        )

        valid = degree > 0

        inv_sqrt[valid] = (
            1.0 / np.sqrt(degree[valid])
        )

        D_inv = sparse.diags(
            inv_sqrt,
            format="csr",
        )

        self.A = (
            D_inv
            @ self.A
            @ D_inv
        ).tocsr()

        self.A.sum_duplicates()

        print(
            f"Global normalized sparse spatial graph: "
            f"{self.A.nnz:,} directed edges."
        )

    def k_hop_nodes(
        self,
        core_nodes: np.ndarray,
        hops: int,
    ) -> np.ndarray:
        """
        Expand core nodes by K graph hops using CSR neighbor lookup.
        """
        selected = set(
            map(int, core_nodes)
        )

        frontier = set(
            selected
        )

        for _ in range(hops):
            next_frontier = set()

            for node in frontier:
                start = self.A.indptr[node]
                end = self.A.indptr[node + 1]

                neigh = self.A.indices[
                    start:end
                ]

                next_frontier.update(
                    map(int, neigh)
                )

            next_frontier -= selected

            if not next_frontier:
                break

            selected.update(
                next_frontier
            )

            frontier = next_frontier

        return np.array(
            sorted(selected),
            dtype=np.int64,
        )

    def induced_subgraph(
        self,
        node_ids: np.ndarray,
    ) -> sparse.csr_matrix:
        """
        Extract A[node_ids, node_ids].
        """
        return self.A[
            node_ids
        ][:, node_ids].tocsr()


# =============================================================================
# 8. SUBGRAPH NORMALIZATION / TORCH COO
# =============================================================================

def scipy_csr_to_torch_sparse(
    A: sparse.csr_matrix,
) -> torch.Tensor:

    A = A.tocoo()

    row = torch.from_numpy(
        A.row.astype(np.int64, copy=False)
    )
    col = torch.from_numpy(
        A.col.astype(np.int64, copy=False)
    )
    val = torch.from_numpy(
        A.data.astype(np.float32, copy=False)
    )

    indices = torch.stack(
        [row, col],
        dim=0,
    )

    return torch.sparse_coo_tensor(
        indices,
        val,
        size=A.shape,
        dtype=torch.float32,
    ).coalesce()


# =============================================================================
# 9. CALENDAR CONTEXT
# =============================================================================

def context_from_date(
    date: pd.Timestamp,
) -> Dict[str, int]:
    dow = int(
        date.dayofweek
    ) + 1

    weekend = int(
        dow in (6, 7)
    )

    if date.day <= 10:
        month_period = 0

    elif date.day <= 20:
        month_period = 1

    else:
        month_period = 2

    return {
        "day_of_week": dow,
        "is_weekend": weekend,
        "month": int(date.month),
        "month_period": month_period,
    }


# =============================================================================
# 10. TEMPORAL SPLITS
# =============================================================================

def make_temporal_splits(
    dates: List[pd.Timestamp],
    cfg: Config,
):
    max_history = max(
        max(cfg.RECENT_LAGS),
        max(cfg.WEEKLY_LAGS),
        max(cfg.LONG_WINDOWS),
    )

    train_end = pd.Timestamp(
        cfg.TRAIN_END
    )

    val_end = pd.Timestamp(
        cfg.VAL_END
    )

    test_end = pd.Timestamp(
        cfg.TEST_END
    )

    train = []
    val = []
    test = []

    for t, d in enumerate(
        dates
    ):
        if t < max_history:
            continue

        if d <= train_end:
            train.append(t)

        elif d <= val_end:
            val.append(t)

        elif d <= test_end:
            test.append(t)

    return (
        train,
        val,
        test,
    )


# =============================================================================
# 11. CORE NODE CHUNKS
# =============================================================================

def build_core_node_chunks(
    nodes: np.ndarray,
    chunk_size: int,
    grid_meta: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> List[np.ndarray]:

    nodes = np.asarray(nodes, dtype=np.int64)

    if (
        grid_meta is not None
        and cfg is not None
        and cfg.SPATIALLY_COHERENT_CHUNKS
    ):
        rows = pd.to_numeric(
            grid_meta[cfg.GRID_ROW_COL],
            errors="coerce",
        ).to_numpy()

        cols = pd.to_numeric(
            grid_meta[cfg.GRID_COL_COL],
            errors="coerce",
        ).to_numpy()

        if np.all(np.isfinite(rows[nodes])) and np.all(np.isfinite(cols[nodes])):
            order = np.lexsort((cols[nodes], rows[nodes]))
            nodes = nodes[order]

    return [
        nodes[start : start + chunk_size].astype(np.int64, copy=False)
        for start in range(0, len(nodes), chunk_size)
    ]


def build_spatial_chunk_cache(
    spatial_graph: SparseSpatialGraph,
    core_chunks: List[np.ndarray],
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, Any]]:

    cache: List[Dict[str, Any]] = []

    use_gpu_cache = (
        cfg.CACHE_SPATIAL_CHUNKS_ON_GPU
        and device.type == "cuda"
        and cfg.NUM_WORKERS == 0
    )

    print(
        f"Precomputing {len(core_chunks):,} spatial chunks "
        f"(adjacency cache on {'GPU' if use_gpu_cache else 'CPU'})..."
    )

    halo_sizes = []

    for chunk_idx, core_global in enumerate(core_chunks):
        sub_nodes = spatial_graph.k_hop_nodes(core_global, cfg.HALO_HOPS)

        # sub_nodes is sorted, so vectorized search is faster than a Python dict.
        core_local = np.searchsorted(sub_nodes, core_global).astype(np.int64)

        local_A = scipy_csr_to_torch_sparse(
            spatial_graph.induced_subgraph(sub_nodes)
        )

        if use_gpu_cache:
            local_A = local_A.to(device)

        cache.append(
            {
                "sub_nodes": sub_nodes,
                "core_global": core_global,
                "core_local": core_local,
                "spatial_A": local_A,
            }
        )

        halo_sizes.append(len(sub_nodes))

    if halo_sizes:
        print(
            "Cached subgraph size: "
            f"mean={np.mean(halo_sizes):.1f}, "
            f"max={np.max(halo_sizes):,}, "
            f"core={cfg.CORE_NODES_PER_BATCH:,}"
        )

    return cache


# =============================================================================
# 12. SUBGRAPH TEMPORAL DATASET
# =============================================================================

class SubgraphTemporalDataset(Dataset):

    def __init__(
        self,
        store: SparseCrimeStore,
        target_times: List[int],
        chunk_cache: List[Dict[str, Any]],
        cfg: Config,
    ):
        self.store = store
        self.target_times = target_times
        self.chunk_cache = chunk_cache
        self.cfg = cfg

        self.num_chunks = len(chunk_cache)

    def __len__(self):
        return len(self.target_times) * self.num_chunks

    def __getitem__(self, idx):
        time_pos, chunk_idx = divmod(int(idx), self.num_chunks)
        t = self.target_times[time_pos]

        cached = self.chunk_cache[chunk_idx]
        sub_nodes = cached["sub_nodes"]
        core_global = cached["core_global"]
        core_local = cached["core_local"]
        local_A = cached["spatial_A"]

        recent = self.store.get_lag_sequence(
            t,
            self.cfg.RECENT_LAGS,
            sub_nodes,
        )

        weekly = self.store.get_lag_sequence(
            t,
            self.cfg.WEEKLY_LAGS,
            sub_nodes,
        )

        longterm = self.store.get_rolling_means(
            t,
            self.cfg.LONG_WINDOWS,
            sub_nodes,
        )

        target = self.store.get_day_nodes(t, core_global)

        if self.cfg.USE_LOG1P_INPUT:
            # In-place transforms avoid another large allocation.
            np.log1p(recent, out=recent)
            np.log1p(weekly, out=weekly)
            np.log1p(longterm, out=longterm)

        ctx = context_from_date(self.store.dates[t])

        recent_ctx = [
            context_from_date(self.store.dates[t - lag])
            for lag in self.cfg.RECENT_LAGS
        ]

        weekly_ctx = [
            context_from_date(self.store.dates[t - lag])
            for lag in self.cfg.WEEKLY_LAGS
        ]

        return {
            "recent": torch.from_numpy(recent).float(),
            "weekly": torch.from_numpy(weekly).float(),
            "longterm": torch.from_numpy(longterm).float(),
            "target": torch.from_numpy(target).float(),

            "spatial_A": local_A,

            "core_local_idx": torch.from_numpy(core_local).long(),
            "core_global_idx": torch.from_numpy(core_global).long(),
            "target_time_idx": torch.tensor(t, dtype=torch.long),
            "chunk_idx": torch.tensor(chunk_idx, dtype=torch.long),

            "day_of_week": torch.tensor(ctx["day_of_week"], dtype=torch.long),
            "is_weekend": torch.tensor(ctx["is_weekend"], dtype=torch.long),
            "month": torch.tensor(ctx["month"], dtype=torch.long),
            "month_period": torch.tensor(ctx["month_period"], dtype=torch.long),

            "recent_day_of_week": torch.tensor(
                [x["day_of_week"] for x in recent_ctx], dtype=torch.long
            ),
            "recent_is_weekend": torch.tensor(
                [x["is_weekend"] for x in recent_ctx], dtype=torch.long
            ),
            "recent_month": torch.tensor(
                [x["month"] for x in recent_ctx], dtype=torch.long
            ),
            "recent_month_period": torch.tensor(
                [x["month_period"] for x in recent_ctx], dtype=torch.long
            ),

            "weekly_day_of_week": torch.tensor(
                [x["day_of_week"] for x in weekly_ctx], dtype=torch.long
            ),
            "weekly_is_weekend": torch.tensor(
                [x["is_weekend"] for x in weekly_ctx], dtype=torch.long
            ),
            "weekly_month": torch.tensor(
                [x["month"] for x in weekly_ctx], dtype=torch.long
            ),
            "weekly_month_period": torch.tensor(
                [x["month_period"] for x in weekly_ctx], dtype=torch.long
            ),
        }


def single_subgraph_collate(
    batch,
):

    if len(batch) != 1:
        raise ValueError(
            "Use DataLoader batch_size=1 "
            "for variable-size subgraphs."
        )

    return batch[0]


# =============================================================================
# 13. SPARSE GRAPH PROPAGATION
# =============================================================================

def sparse_graph_propagation(
    A: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:

    if not A.is_sparse:
        raise TypeError("A must be sparse COO.")

    if x.dim() == 2:
        return torch.sparse.mm(A, x)

    if x.dim() == 3:
        L, N, Fdim = x.shape

        # [L,N,F] -> [N,L*F]
        packed = (
            x.permute(1, 0, 2)
            .contiguous()
            .reshape(N, L * Fdim)
        )

        propagated = torch.sparse.mm(A, packed)

        return (
            propagated
            .reshape(N, L, Fdim)
            .permute(1, 0, 2)
            .contiguous()
        )

    raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")


class SparseGraphConv(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        x = sparse_graph_propagation(A, x)
        x = self.linear(x)
        x = F.relu(x)
        return self.dropout(x)


class SpatialGraphEncoder(
    nn.Module
):


    def __init__(
        self,
        crime_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()

        blocks = []
        in_dim = crime_dim

        for _ in range(
            layers
        ):
            blocks.append(
                SparseGraphConv(
                    in_dim,
                    hidden_dim,
                    dropout,
                )
            )

            in_dim = (
                hidden_dim
            )

        self.layers = (
            nn.ModuleList(
                blocks
            )
        )

    def forward(
        self,
        x,
        A,
    ):
        h = x

        for layer in (
            self.layers
        ):
            h = layer(
                h,
                A,
            )

        return h


# =============================================================================
# 14. CONTEXT ENCODER
# =============================================================================

class ContextEncoder(
    nn.Module
):
    def __init__(
        self,
        out_dim: int,
    ):
        super().__init__()

        self.dow = nn.Embedding(
            8,
            8,
        )

        self.weekend = (
            nn.Embedding(
                2,
                4,
            )
        )

        self.month = nn.Embedding(
            13,
            8,
        )

        self.month_period = (
            nn.Embedding(
                3,
                4,
            )
        )

        self.project = (
            nn.Sequential(
                nn.Linear(
                    24,
                    out_dim,
                ),
                nn.ReLU(),
            )
        )

    def forward(
        self,
        day_of_week,
        is_weekend,
        month,
        month_period,
    ):

        x = torch.cat(
            [
                self.dow(
                    day_of_week
                ),
                self.weekend(
                    is_weekend
                ),
                self.month(
                    month
                ),
                self.month_period(
                    month_period
                ),
            ],
            dim=-1,
        )

        return self.project(
            x
        )


# =============================================================================
# 15. CONTEXT-ADAPTIVE CRIME GRAPH
# =============================================================================

class ContextAdaptiveCrimeGraph(
    nn.Module
):


    def __init__(
        self,
        num_crimes: int,
        embed_dim: int,
        context_dim: int,
        use_context: bool,
        use_state: bool,
        use_topk: bool,
        topk: int,
    ):
        super().__init__()

        self.C = num_crimes
        self.use_context = use_context
        self.use_state = use_state
        self.use_topk = use_topk
        self.topk = topk

        self.crime_embedding = nn.Parameter(
            torch.randn(
                num_crimes,
                embed_dim,
            ) * 0.02
        )

        self.q = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.k = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        if use_context:
            self.context_bias = nn.Sequential(
                nn.Linear(
                    context_dim,
                    context_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    context_dim,
                    num_crimes * num_crimes,
                ),
            )

        if use_state:
            self.state_bias = nn.Sequential(
                nn.Linear(
                    num_crimes,
                    max(16, num_crimes * 2),
                ),
                nn.ReLU(),
                nn.Linear(
                    max(16, num_crimes * 2),
                    num_crimes * num_crimes,
                ),
            )

    def forward(
        self,
        context: torch.Tensor,
        snapshot_state: torch.Tensor,
    ):

        Q = self.q(self.crime_embedding)
        K = self.k(self.crime_embedding)

        base = (Q @ K.T) / math.sqrt(Q.shape[-1])

        batched = context.dim() == 2

        if batched:
            L = context.shape[0]
            logits = base.unsqueeze(0).expand(L, -1, -1)
        else:
            logits = base

        if self.use_context:
            cb = self.context_bias(context)
            if batched:
                cb = cb.view(-1, self.C, self.C)
            else:
                cb = cb.view(self.C, self.C)
            logits = logits + cb

        if self.use_state:
            sb = self.state_bias(snapshot_state)
            if batched:
                sb = sb.view(-1, self.C, self.C)
            else:
                sb = sb.view(self.C, self.C)

            logits = logits + sb
        A = torch.softmax(logits, dim=-1)

        if self.use_topk:
            k = min(self.topk, self.C)
            _, idx = torch.topk(A, k=k, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(-1, idx, 1.0)
            A = A * mask
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return A


class CrimeGraphEncoder(
    nn.Module
):


    def __init__(
        self,
        num_crimes: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.C = (
            num_crimes
        )

        self.input_proj = (
            nn.Linear(
                1,
                hidden_dim,
            )
        )

        self.output_proj = (
            nn.Linear(
                num_crimes
                * hidden_dim,
                hidden_dim,
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        x,
        A_crime,
    ):
        # 2D snapshot: [N,C] -> [N,C,H]
        if x.dim() == 2:
            h = self.input_proj(x.unsqueeze(-1))
            h = torch.einsum("cd,ndh->nch", A_crime, h)
            h = F.relu(h)
            N, C, H = h.shape
            h = h.reshape(N, C * H)
            h = F.relu(self.output_proj(h))
            return self.dropout(h)

        # Vectorized sequence: [L,N,C] -> [L,N,C,H]
        if x.dim() == 3:
            h = self.input_proj(x.unsqueeze(-1))
            h = torch.einsum("lcd,lndh->lnch", A_crime, h)
            h = F.relu(h)
            L, N, C, H = h.shape
            h = h.reshape(L, N, C * H)
            h = F.relu(self.output_proj(h))
            return self.dropout(h)

        raise ValueError(f"Unsupported crime input shape: {tuple(x.shape)}")


# =============================================================================
# 16. DUAL-GRAPH FUSION
# =============================================================================

class DualGraphFusion(
    nn.Module
):


    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        gated: bool,
        dropout: float,
    ):
        super().__init__()

        self.gated = (
            gated
        )

        total_dim = (
            hidden_dim * 2
            + context_dim
        )

        if gated:
            self.gate = (
                nn.Sequential(
                    nn.Linear(
                        total_dim,
                        hidden_dim,
                    ),
                    nn.Sigmoid(),
                )
            )

        else:
            self.project = (
                nn.Sequential(
                    nn.Linear(
                        total_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        dropout
                    ),
                )
            )

    def forward(
        self,
        Hs,
        Hc,
        context,
    ):
        if Hs.dim() == 2:
            N = Hs.shape[0]
            ctx = context.unsqueeze(0).expand(N, -1)

        elif Hs.dim() == 3:
            L, N, _ = Hs.shape
            if context.dim() == 1:
                context = context.unsqueeze(0).expand(L, -1)

            ctx = context.unsqueeze(1).expand(-1, N, -1)

        else:
            raise ValueError(f"Unsupported fusion shape: {tuple(Hs.shape)}")

        z = torch.cat([Hs, Hc, ctx], dim=-1)

        if self.gated:
            g = self.gate(z)
            h = g * Hs + (1.0 - g) * Hc
            return h, g

        return self.project(z), None


# =============================================================================
# 17. TEMPORAL ENCODER
# =============================================================================

class TemporalTCN(
    nn.Module
):


    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        x,
    ):
        # [L,N,H] -> [N,H,L]
        x = (
            x.permute(
                1,
                2,
                0,
            )
            .contiguous()
        )

        x = (
            self.dropout(
                F.relu(
                    self.conv1(
                        x
                    )
                )
            )
        )

        x = F.relu(
            self.conv2(
                x
            )
        )

        # Last temporal position.
        return x[
            :,
            :,
            -1,
        ]


class LongTermScaleEncoder(
    nn.Module
):


    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.project = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        scale_features: torch.Tensor,
    ):
        # [S,N,H] -> [N,S,H]
        x = scale_features.permute(
            1,
            0,
            2,
        ).contiguous()

        scores = self.score(
            x
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        h = (
            x
            * weights.unsqueeze(-1)
        ).sum(
            dim=1
        )

        return (
            self.project(h),
            weights,
        )


class TemporalStreamAttention(
    nn.Module
):


    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()

        self.score = (
            nn.Sequential(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
                nn.Tanh(),
                nn.Linear(
                    hidden_dim,
                    1,
                ),
            )
        )

    def forward(
        self,
        streams,
    ):
        x = torch.stack(
            streams,
            dim=1,
        )
        # [N,S,H]

        scores = (
            self.score(
                x
            )
            .squeeze(-1)
        )

        weights = (
            torch.softmax(
                scores,
                dim=-1,
            )
        )

        h = (
            x
            * weights.unsqueeze(
                -1
            )
        ).sum(
            dim=1
        )

        return (
            h,
            weights,
        )


# =============================================================================
# 18. PROPOSED-ST MODEL
# =============================================================================

class ProposedST(
    nn.Module
):


    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()

        self.cfg = cfg

        self.C = len(
            cfg.CRIME_TYPES
        )

        self.context_encoder = (
            ContextEncoder(
                cfg.CONTEXT_DIM
            )
        )

        self.null_context = (
            nn.Parameter(
                torch.zeros(
                    cfg.CONTEXT_DIM
                )
            )
        )

        self.raw_project = (
            nn.Linear(
                self.C,
                cfg.HIDDEN_DIM,
            )
        )

        self.spatial_encoder = (
            SpatialGraphEncoder(
                crime_dim=self.C,
                hidden_dim=cfg.HIDDEN_DIM,
                layers=cfg.NUM_GCN_LAYERS,
                dropout=cfg.DROPOUT,
            )
        )

        self.crime_graph = (
            ContextAdaptiveCrimeGraph(
                num_crimes=self.C,
                embed_dim=cfg.CRIME_EMBED_DIM,
                context_dim=cfg.CONTEXT_DIM,
                use_context=(
                    cfg.USE_CONTEXT_ADAPTIVE_CRIME_GRAPH
                ),
                use_state=(
                    cfg.USE_STATE_ADAPTIVE_CRIME_GRAPH
                ),
                use_topk=(
                    cfg.USE_TOPK_CRIME_GRAPH
                ),
                topk=(
                    cfg.CRIME_GRAPH_TOPK
                ),
            )
        )

        self.crime_encoder = (
            CrimeGraphEncoder(
                num_crimes=self.C,
                hidden_dim=cfg.HIDDEN_DIM,
                dropout=cfg.DROPOUT,
            )
        )

        self.fusion = (
            DualGraphFusion(
                hidden_dim=cfg.HIDDEN_DIM,
                context_dim=cfg.CONTEXT_DIM,
                gated=cfg.USE_GATED_FUSION,
                dropout=cfg.DROPOUT,
            )
        )

        self.recent_tcn = (
            TemporalTCN(
                cfg.HIDDEN_DIM,
                cfg.DROPOUT,
            )
        )

        self.weekly_tcn = (
            TemporalTCN(
                cfg.HIDDEN_DIM,
                cfg.DROPOUT,
            )
        )

        self.long_scale_encoder = (
            LongTermScaleEncoder(
                cfg.HIDDEN_DIM,
                cfg.DROPOUT,
            )
        )

        self.temporal_attention = (
            TemporalStreamAttention(
                cfg.HIDDEN_DIM
            )
        )

        self.output_head = (
            nn.Sequential(
                nn.Linear(
                    cfg.HIDDEN_DIM,
                    cfg.HIDDEN_DIM,
                ),
                nn.ReLU(),
                nn.Dropout(
                    cfg.DROPOUT
                ),
                nn.Linear(
                    cfg.HIDDEN_DIM,
                    self.C,
                ),
                # Positive count/intensity mean.
                nn.Softplus(),
            )
        )

    def encode_context(
        self,
        batch,
    ):

        if self.cfg.USE_CONTEXT_FEATURES:
            return self.context_encoder(
                batch["day_of_week"],
                batch["is_weekend"],
                batch["month"],
                batch["month_period"],
            )

        return self.null_context

    def encode_context_sequence(
        self,
        batch,
        prefix: str,
    ) -> torch.Tensor:

        if not self.cfg.USE_CONTEXT_FEATURES:
            L = batch[
                f"{prefix}_day_of_week"
            ].shape[0]

            return (
                self.null_context
                .unsqueeze(0)
                .expand(
                    L,
                    -1,
                )
            )

        return self.context_encoder(
            batch[
                f"{prefix}_day_of_week"
            ],
            batch[
                f"{prefix}_is_weekend"
            ],
            batch[
                f"{prefix}_month"
            ],
            batch[
                f"{prefix}_month_period"
            ],
        )

    def encode_snapshot(
        self,
        x,
        spatial_A,
        context,
        collect_explanations: bool = False,
    ):

        explanations = {}

        if self.cfg.USE_SPATIAL_GRAPH:
            Hs = self.spatial_encoder(x, spatial_A)
        else:
            Hs = F.relu(self.raw_project(x))

        if self.cfg.USE_CRIME_GRAPH:
            # Mean over node dimension. Works for [N,C] and [L,N,C].
            snapshot_state = x.mean(dim=-2)

            Ac = self.crime_graph(context, snapshot_state)
            Hc = self.crime_encoder(x, Ac)

            if collect_explanations:
                explanations["crime_adjacency"] = Ac
        else:
            Hc = F.relu(self.raw_project(x))

        if self.cfg.USE_SPATIAL_GRAPH and self.cfg.USE_CRIME_GRAPH:
            h, gate = self.fusion(Hs, Hc, context)

            if collect_explanations and gate is not None:
                explanations["fusion_gate"] = gate

        elif self.cfg.USE_SPATIAL_GRAPH:
            h = Hs
        elif self.cfg.USE_CRIME_GRAPH:
            h = Hc
        else:
            h = F.relu(self.raw_project(x))

        return h, explanations

    def encode_sequence(
        self,
        sequence,
        spatial_A,
        contexts,
        collect_explanations: bool = False,
    ):

        h, exp = self.encode_snapshot(
            sequence,
            spatial_A,
            contexts,
            collect_explanations=collect_explanations,
        )

        result = {"sequence": h}
        result.update(exp)
        return result

    def forward(
        self,
        batch,
        return_explanations: bool = False,
    ):
        context = self.encode_context(batch)

        recent_contexts = self.encode_context_sequence(batch, "recent")
        weekly_contexts = self.encode_context_sequence(batch, "weekly")

        spatial_A = batch["spatial_A"]

        streams = []
        names = []
        explanations = {}

        if self.cfg.USE_RECENT_STREAM:
            z = self.encode_sequence(
                batch["recent"],
                spatial_A,
                recent_contexts,
                collect_explanations=return_explanations,
            )

            streams.append(self.recent_tcn(z["sequence"]))
            names.append("recent")

            if return_explanations:
                for k, v in z.items():
                    if k != "sequence":
                        explanations[f"recent_{k}"] = v

        if self.cfg.USE_WEEKLY_STREAM:
            z = self.encode_sequence(
                batch["weekly"],
                spatial_A,
                weekly_contexts,
                collect_explanations=return_explanations,
            )

            streams.append(self.weekly_tcn(z["sequence"]))
            names.append("weekly")

            if return_explanations:
                for k, v in z.items():
                    if k != "sequence":
                        explanations[f"weekly_{k}"] = v

        if self.cfg.USE_LONGTERM_STREAM:
            S = batch["longterm"].shape[0]
            scale_context = context.unsqueeze(0).expand(S, -1)

            scale_encoded, scale_exp = self.encode_snapshot(
                batch["longterm"],
                spatial_A,
                scale_context,
                collect_explanations=return_explanations,
            )

            long_h, scale_weights = self.long_scale_encoder(scale_encoded)
            streams.append(long_h)
            names.append("longterm")

            if return_explanations:
                explanations["longterm_scale_weights"] = scale_weights
                explanations["longterm_scale_windows"] = list(
                    self.cfg.LONG_WINDOWS
                )

                for k, v in scale_exp.items():
                    explanations[f"longterm_{k}"] = v

        if not streams:
            raise ValueError("At least one temporal stream must be enabled.")

        if self.cfg.USE_TEMPORAL_ATTENTION and len(streams) > 1:
            h, weights = self.temporal_attention(streams)

            if return_explanations:
                explanations["temporal_stream_weights"] = weights
                explanations["temporal_stream_names"] = names
        else:
            h = torch.stack(streams, dim=0).mean(dim=0)

        pred_all = self.output_head(h)
        pred_core = pred_all[batch["core_local_idx"]]

        return pred_core, explanations


# =============================================================================
# 19. DEVICE MOVEMENT
# =============================================================================

def move_batch_to_device(
    batch,
    device,
):

    result = {}

    for k, v in (
        batch.items()
    ):
        if torch.is_tensor(
            v
        ):
            result[k] = (
                v.to(
                    device
                )
            )

        else:
            result[k] = v

    return result


def single_task_count_loss(
    pred_mean: torch.Tensor,
    target_count: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:

    name = cfg.LOSS_NAME.lower()

    if name == "poisson":
        return F.poisson_nll_loss(
            pred_mean.clamp_min(1e-8),
            target_count,
            log_input=False,
            full=True,
            reduction="mean",
        )

    if name == "mae":
        return F.l1_loss(
            pred_mean,
            target_count,
        )

    if name == "huber":
        return F.huber_loss(
            pred_mean,
            target_count,
            delta=cfg.HUBER_DELTA,
            reduction="mean",
        )

    raise ValueError(
        f"Unsupported LOSS_NAME={cfg.LOSS_NAME!r}"
    )



# =============================================================================
# 20. TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
):
    model.train()

    total = 0.0
    count = 0
    start_time = time.perf_counter()

    amp_enabled = (
        model.cfg.USE_AMP
        and device.type == "cuda"
    )

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            pred, _ = model(
                batch,
                return_explanations=False,
            )

            target = batch["target"]

            loss = single_task_count_loss(
                pred,
                target,
                cfg=model.cfg,
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        n_core = target.shape[0]
        total += loss.item() * n_core
        count += n_core

    elapsed = time.perf_counter() - start_time

    return (
        total / max(count, 1),
        elapsed,
    )


# =============================================================================
# 21. EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    cfg,
    max_batches=None,
):

    model.eval()

    sum_abs = 0.0
    sum_sq = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n_values = 0

    poisson_sum = 0.0
    poisson_n = 0

    ranking_rows: List[Dict[str, float]] = []

    current_t: Optional[int] = None

    N = loader.dataset.store.N

    current_pred_total = np.zeros(N, dtype=np.float32)
    current_true_total = np.zeros(N, dtype=np.float32)
    current_seen = np.zeros(N, dtype=bool)

    def finalize_current_day():
        if current_t is None:
            return

        mask = current_seen

        if not np.any(mask):
            return

        ranking_rows.append(
            hotspot_metrics_for_one_day(
                current_pred_total[mask],
                current_true_total[mask],
                cfg.HOTSPOT_FRACTIONS,
            )
        )

        current_pred_total.fill(0.0)
        current_true_total.fill(0.0)
        current_seen.fill(False)

    for batch_idx, batch in enumerate(loader):
        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        target_t = int(
            batch["target_time_idx"].item()
        )

        if (
            current_t is not None
            and target_t != current_t
        ):
            finalize_current_day()

        current_t = target_t

        core_global_idx = (
            batch["core_global_idx"]
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )

        batch = move_batch_to_device(
            batch,
            device,
        )

        pred, _ = model(batch, return_explanations=False)
        target = batch["target"]

        # v3 predicts directly on the original non-negative count scale.
        pred = pred.clamp_min(0.0)
        target = target.clamp_min(0.0)

        diff = pred - target

        sum_abs += torch.abs(diff).sum().item()
        sum_sq += (diff ** 2).sum().item()
        sum_y += target.sum().item()
        sum_y2 += (target ** 2).sum().item()
        n_values += diff.numel()

        pred_np = (
            pred.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        target_np = (
            target.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        poisson_sum += (
            poisson_deviance_from_arrays(
                pred_np,
                target_np,
            )
            * pred_np.size
        )

        poisson_n += pred_np.size

        current_pred_total[
            core_global_idx
        ] = pred_np.sum(axis=1)

        current_true_total[
            core_global_idx
        ] = target_np.sum(axis=1)

        current_seen[
            core_global_idx
        ] = True

    finalize_current_day()

    metrics = {
        "MAE": sum_abs / max(n_values, 1),
        "RMSE": math.sqrt(
            sum_sq / max(n_values, 1)
        ),
        "R2": r2_from_sufficient_statistics(
            sum_y,
            sum_y2,
            sum_sq,
            n_values,
        ),
        "PoissonDeviance": (
            poisson_sum / max(poisson_n, 1)
        ),
    }

    metrics.update(
        average_metric_dicts(ranking_rows)
    )

    return metrics


# =============================================================================
# 22. EXPLANATION EXPORT
# =============================================================================

@torch.no_grad()
def export_explanation_sample(
    model,
    loader,
    device,
    output_dir,
):
    model.eval()

    batch = next(
        iter(
            loader
        )
    )

    batch = (
        move_batch_to_device(
            batch,
            device,
        )
    )

    _, exp = model(
        batch,
        return_explanations=True,
    )

    output = {}

    for k, v in (
        exp.items()
    ):
        if torch.is_tensor(
            v
        ):
            # Save only compact summaries for large node-wise tensors.
            arr = (
                v.detach()
                .cpu()
            )

            if arr.numel() <= 5000:
                output[k] = (
                    arr.numpy()
                    .tolist()
                )
            else:
                output[k] = {
                    "shape":
                        list(
                            arr.shape
                        ),
                    "mean":
                        float(
                            arr.float()
                            .mean()
                        ),
                    "std":
                        float(
                            arr.float()
                            .std()
                        ),
                    "min":
                        float(
                            arr.float()
                            .min()
                        ),
                    "max":
                        float(
                            arr.float()
                            .max()
                        ),
                }

        else:
            output[k] = v

    with open(
        Path(output_dir)
        / "explanations_sample.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# 23. LOGGING AND PLOTTING
# =============================================================================

def append_text_log(
    log_path: Path,
    message: str,
) -> None:
    """Append one line to a human-readable experiment log."""
    with open(
        log_path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(message.rstrip() + "\n")


def save_history_plots(
    history_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save loss/validation metric line charts."""
    if history_df.empty:
        return

    # 1) Training convergence
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train loss",
    )

    if "val_MAE" in history_df.columns:
        plt.plot(
            history_df["epoch"],
            history_df["val_MAE"],
            label="Validation MAE",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss / error")
    plt.title("Training convergence")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig.savefig(
        output_dir / "loss_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 2) Validation point metrics
    point_cols = [
        c
        for c in [
            "val_MAE",
            "val_RMSE",
            "val_PoissonDeviance",
        ]
        if c in history_df.columns
    ]

    if point_cols:
        fig = plt.figure(figsize=(8, 5))

        for col in point_cols:
            plt.plot(
                history_df["epoch"],
                history_df[col],
                label=col.replace("val_", ""),
            )

        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title("Validation point-forecast metrics")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        fig.savefig(
            output_dir / "validation_point_metrics.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    # 3) Validation hotspot metrics
    hotspot_cols = [
        c
        for c in history_df.columns
        if (
            c.startswith("val_PAI@")
            or c.startswith("val_HitRate@")
            or c.startswith("val_NDCG@")
        )
    ]

    if hotspot_cols:
        fig = plt.figure(figsize=(9, 5))

        for col in hotspot_cols:
            plt.plot(
                history_df["epoch"],
                history_df[col],
                label=col.replace("val_", ""),
            )

        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.title("Validation hotspot-ranking metrics")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        fig.savefig(
            output_dir / "validation_hotspot_metrics.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)


def save_test_metric_plots(
    metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    """Save final test metric bar charts."""
    point_keys = [
        k
        for k in [
            "MAE",
            "RMSE",
            "PoissonDeviance",
        ]
        if k in metrics
    ]

    if point_keys:
        fig = plt.figure(figsize=(7, 4.5))

        plt.bar(
            point_keys,
            [metrics[k] for k in point_keys],
        )

        plt.ylabel("Value")
        plt.title("Test point-forecast metrics")
        plt.tight_layout()

        fig.savefig(
            output_dir / "test_point_metrics.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    groups = {
        "PAI": [
            k for k in metrics
            if k.startswith("PAI@")
        ],
        "HitRate": [
            k for k in metrics
            if k.startswith("HitRate@")
        ],
        "NDCG": [
            k for k in metrics
            if k.startswith("NDCG@")
        ],
    }

    for group, keys in groups.items():
        if not keys:
            continue

        keys = sorted(
            keys,
            key=lambda x: int(
                x.split("@")[1].replace("%", "")
            ),
        )

        fig = plt.figure(figsize=(7, 4.5))

        plt.bar(
            keys,
            [metrics[k] for k in keys],
        )

        plt.ylabel(group)
        plt.title(f"Test {group} hotspot metrics")
        plt.tight_layout()

        fig.savefig(
            output_dir / f"test_{group.lower()}_metrics.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)



# =============================================================================
# 23. ABLATION PRESETS
# =============================================================================

def apply_ablation_preset(
    cfg: Config,
    name: str,
) -> Config:

    cfg = Config(
        **asdict(
            cfg
        )
    )

    name = name.upper()

    if name == "FULL":
        pass

    elif name == (
        "SPATIAL_ONLY"
    ):
        cfg.USE_SPATIAL_GRAPH = (
            True
        )
        cfg.USE_CRIME_GRAPH = (
            False
        )

    elif name == (
        "CRIME_ONLY"
    ):
        cfg.USE_SPATIAL_GRAPH = (
            False
        )
        cfg.USE_CRIME_GRAPH = (
            True
        )

    elif name == (
        "STATIC_CRIME_GRAPH"
    ):
        cfg.USE_CRIME_GRAPH = (
            True
        )

        cfg.USE_CONTEXT_ADAPTIVE_CRIME_GRAPH = (
            False
        )
        cfg.USE_STATE_ADAPTIVE_CRIME_GRAPH = (
            False
        )

    elif name == (
        "NO_CONTEXT"
    ):
        cfg.USE_CONTEXT_FEATURES = (
            False
        )

        cfg.USE_CONTEXT_ADAPTIVE_CRIME_GRAPH = (
            False
        )

    elif name == (
        "RECENT_ONLY"
    ):
        cfg.USE_RECENT_STREAM = (
            True
        )
        cfg.USE_WEEKLY_STREAM = (
            False
        )
        cfg.USE_LONGTERM_STREAM = (
            False
        )

    elif name == (
        "NO_WEEKLY"
    ):
        cfg.USE_WEEKLY_STREAM = (
            False
        )

    elif name == (
        "NO_LONGTERM"
    ):
        cfg.USE_LONGTERM_STREAM = (
            False
        )

    elif name == (
        "NO_TEMPORAL_ATTENTION"
    ):
        cfg.USE_TEMPORAL_ATTENTION = (
            False
        )

    elif name == (
        "NO_GATED_FUSION"
    ):
        cfg.USE_GATED_FUSION = (
            False
        )

    elif name == (
        "DENSE_CRIME_GRAPH"
    ):
        cfg.USE_TOPK_CRIME_GRAPH = (
            False
        )

    else:
        raise ValueError(
            f"Unknown ablation preset: "
            f"{name}"
        )

    return cfg


# =============================================================================
# 24. MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    ablation_name: str = "FULL",
):
    cfg = (
        apply_ablation_preset(
            CFG,
            ablation_name,
        )
    )

    # Temporal ordering must be oldest -> newest because TemporalTCN returns
    # the final temporal position.
    if tuple(cfg.RECENT_LAGS) != tuple(
        sorted(cfg.RECENT_LAGS, reverse=True)
    ):
        raise ValueError(
            "RECENT_LAGS must be ordered oldest -> newest, "
            "e.g. (7,6,5,4,3,2,1)."
        )

    if tuple(cfg.WEEKLY_LAGS) != tuple(
        sorted(cfg.WEEKLY_LAGS, reverse=True)
    ):
        raise ValueError(
            "WEEKLY_LAGS must be ordered oldest -> newest, "
            "e.g. (28,21,14,7)."
        )

    # Halo should cover GCN receptive field.
    if (
        cfg.USE_SPATIAL_GRAPH
        and cfg.HALO_HOPS
        < cfg.NUM_GCN_LAYERS
    ):
        raise ValueError(
            "HALO_HOPS should be >= NUM_GCN_LAYERS "
            "to preserve the spatial GCN receptive field "
            "for core nodes."
        )

    run_dir = (
        Path(
            cfg.OUTPUT_DIR
        )
        / ablation_name.lower()
    )

    cfg.OUTPUT_DIR = str(
        run_dir
    )

    ensure_dir(
        cfg.OUTPUT_DIR
    )

    log_path = run_dir / "run.log"

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Ablation: {ablation_name}\n"
        )

    with open(
        run_dir
        / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(
                cfg
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Ablation: "
        f"{ablation_name}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    append_text_log(
        log_path,
        f"Device: {device}",
    )

    # -------------------------------------------------------------------------
    # Long-format sparse data
    # -------------------------------------------------------------------------
    df = (
        load_and_standardize_data(
            cfg
        )
    )

    # -------------------------------------------------------------------------
    # Sparse temporal store
    # -------------------------------------------------------------------------
    store = (
        SparseCrimeStore(
            df,
            cfg,
        )
    )

    # The source DataFrame is no longer needed after the memmap cache exists.
    del df
    gc.collect()

    # -------------------------------------------------------------------------
    # Grid metadata + global sparse graph
    # -------------------------------------------------------------------------
    grid_meta = (
        extract_grid_metadata(
            cfg.DATA_PATH,
            cfg,
            store.grid_ids,
        )
    )

    spatial_graph = (
        SparseSpatialGraph(
            grid_meta,
            cfg,
        )
    )

    # -------------------------------------------------------------------------
    # Temporal split
    # -------------------------------------------------------------------------
    (
        train_times,
        val_times,
        test_times,
    ) = make_temporal_splits(
        store.dates,
        cfg,
    )

    print(
        "Target dates — "
        f"train={len(train_times):,}, "
        f"val={len(val_times):,}, "
        f"test={len(test_times):,}"
    )

    if (
        not train_times
        or not val_times
        or not test_times
    ):
        raise ValueError(
            "One temporal split is empty. "
            "Check TRAIN_END / VAL_END / TEST_END "
            "and the input date range."
        )

    # -------------------------------------------------------------------------
    # Core target nodes
    # -------------------------------------------------------------------------
    all_nodes = np.arange(
        store.N,
        dtype=np.int64,
    )

    if (
        cfg.TRAIN_ACTIVE_NODES_ONLY
    ):
        train_nodes = (
            store.active_training_nodes(
                max(
                    train_times
                )
            )
        )

        print(
            f"Training on active nodes only: "
            f"{len(train_nodes):,}/"
            f"{store.N:,}"
        )

    else:
        train_nodes = (
            all_nodes
        )

    train_chunks = build_core_node_chunks(
        train_nodes,
        cfg.CORE_NODES_PER_BATCH,
        grid_meta=grid_meta,
        cfg=cfg,
    )

    eval_chunks = build_core_node_chunks(
        all_nodes,
        cfg.CORE_NODES_PER_BATCH,
        grid_meta=grid_meta,
        cfg=cfg,
    )

    # Spatial structures are invariant over dates: compute once.
    train_chunk_cache = build_spatial_chunk_cache(
        spatial_graph,
        train_chunks,
        cfg,
        device,
    )

    same_chunks = (
        len(train_chunks) == len(eval_chunks)
        and all(
            np.array_equal(a, b)
            for a, b in zip(train_chunks, eval_chunks)
        )
    )

    if same_chunks:
        eval_chunk_cache = train_chunk_cache
    else:
        eval_chunk_cache = build_spatial_chunk_cache(
            spatial_graph,
            eval_chunks,
            cfg,
            device,
        )

    print(
        f"Core-node chunks: "
        f"train={len(train_chunks):,}, "
        f"eval={len(eval_chunks):,}"
    )

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    train_ds = SubgraphTemporalDataset(
        store,
        train_times,
        train_chunk_cache,
        cfg,
    )

    val_ds = SubgraphTemporalDataset(
        store,
        val_times,
        eval_chunk_cache,
        cfg,
    )

    test_ds = SubgraphTemporalDataset(
        store,
        test_times,
        eval_chunk_cache,
        cfg,
    )

    print(
        f"Subgraph samples: "
        f"train={len(train_ds):,}, "
        f"val={len(val_ds):,}, "
        f"test={len(test_ds):,}"
    )

    # Variable-size subgraphs => batch_size=1.
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=(
            single_subgraph_collate
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=(
            single_subgraph_collate
        ),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=(
            single_subgraph_collate
        ),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = ProposedST(
        cfg
    ).to(
        device
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=(
                cfg.LEARNING_RATE
            ),
            weight_decay=(
                cfg.WEIGHT_DECAY
            ),
        )
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                cfg.USE_AMP
                and device.type == "cuda"
            ),
        )
    except Exception:

        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                cfg.USE_AMP
                and device.type == "cuda"
            )
        )

    best_val = float(
        "inf"
    )

    patience = 0
    history = []

    best_path = (
        run_dir
        / "best_model.pt"
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
        )

        do_validation = (
            epoch == 1
            or epoch % cfg.VAL_EVERY == 0
            or epoch == cfg.EPOCHS
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
        }

        if do_validation:
            val_start = time.perf_counter()

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                cfg,
                max_batches=cfg.MAX_VAL_BATCHES,
            )

            val_seconds = time.perf_counter() - val_start

            history_row["val_seconds"] = val_seconds
            history_row.update(
                {f"val_{k}": v for k, v in val_metrics.items()}
            )

            metric_text = " | ".join(
                [f"val_{k}={v:.6f}" for k, v in val_metrics.items()]
            )

            epoch_message = (
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"train_time={train_seconds/3600:.2f}h | "
                f"val_time={val_seconds/3600:.2f}h | "
                + metric_text
            )

            if val_metrics["MAE"] < best_val:
                best_val = val_metrics["MAE"]
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1

        else:
            val_metrics = None

            epoch_message = (
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"train_time={train_seconds/3600:.2f}h | "
                "validation=skipped"
            )

        history.append(history_row)

        print(epoch_message)
        append_text_log(log_path, epoch_message)

        total_epoch_seconds = time.perf_counter() - epoch_start

        if cfg.PROFILE_TIMING:
            timing_message = (
                f"Epoch {epoch:03d} total time: "
                f"{total_epoch_seconds/3600:.2f} h"
            )
            print(timing_message)
            append_text_log(log_path, timing_message)

        # Patience counts validation checks, not raw epochs.
        if do_validation and patience >= cfg.EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            append_text_log(log_path, "Early stopping.")
            break

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    history_df.to_json(
        run_dir / "training_history.jsonl",
        orient="records",
        lines=True,
    )

    if cfg.SAVE_PLOTS:
        save_history_plots(
            history_df,
            run_dir,
        )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------
    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
        )
    )

    test_metrics = (
        evaluate(
            model,
            test_loader,
            device,
            cfg,
            max_batches=(
                cfg.MAX_TEST_BATCHES
            ),
        )
    )

    print(
        "Test metrics:",
        test_metrics,
    )

    append_text_log(
        log_path,
        "Test metrics: "
        + json.dumps(
            test_metrics,
            ensure_ascii=False,
        ),
    )

    with open(
        run_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
        )

    if cfg.SAVE_PLOTS:
        save_test_metric_plots(
            test_metrics,
            run_dir,
        )

    # Save the best model currently loaded after validation-based selection.
    torch.save(
        model.state_dict(),
        run_dir / "final_model_state_dict.pt",
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "config":
                asdict(cfg),
            "ablation_name":
                ablation_name,
            "best_val_MAE":
                best_val,
            "test_metrics":
                test_metrics,
            "crime_types":
                list(cfg.CRIME_TYPES),
        },
        run_dir / "trained_model_checkpoint.pt",
    )

    # -------------------------------------------------------------------------
    # Explanation snapshot
    # -------------------------------------------------------------------------
    if (
        cfg.RETURN_EXPLANATIONS
        and len(test_ds) > 0
    ):
        export_explanation_sample(
            model,
            test_loader,
            device,
            cfg.OUTPUT_DIR,
        )

if __name__ == "__main__":

    ABLATION_NAME = "FULL"

    run_experiment(
        ablation_name=(
            ABLATION_NAME
        )
    )
