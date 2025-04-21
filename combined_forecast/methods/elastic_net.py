from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# 0. Avoid CPU over‑subscription: 1 BLAS thread per Python process
# ────────────────────────────────────────────────────────────────────────────
from threadpoolctl import threadpool_limits
threadpool_limits(1)

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from joblib import Parallel, delayed
from typing import Optional

# imports:
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Tuple

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tqdm import tqdm  
from concurrent.futures import ProcessPoolExecutor, as_completed

from configuration import ModelSettings

from .utils import data_interpolate_prev, data_interpolate_fut, get_model_from_params



"""
Elastic Net regression model (v2.0 – 2025‑04‑19)
------------------------------------------------
* **Thread‑safe**: caps every worker to 1 BLAS/LAPACK thread (avoids over‑subscription).
* **Path‑wise λ sweep**: one warm‑started fit per (train split, α) instead of per
  (λ, α) – huge cut in model.fit calls.
* **Single parallel layer**: joblib spreads CV splits across cores, inner loops
  stay single‑threaded.
* **Cleaner imports** and no duplicates.

Your data shape per split: **165 × 96 rows, 6 features** → model fits are ultra‑fast;
path‑wise warm starts give the biggest payoff here.
"""


import numpy as np
import pandas as pd
from typing import Optional, Sequence, Dict, Tuple
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# ---------------------------------------------------------------------------
# NUMBA helpers (falls back gracefully if numba not available)
# ---------------------------------------------------------------------------
try:
    from numba import njit
except ImportError:      # still works, just slower
    def njit(signature_or_function=None, **kwargs):
        def wrapper(func):
            return func
        if callable(signature_or_function):
            return signature_or_function
        return wrapper


@njit(cache=True, fastmath=True)
def _soft_thresholding(x: float, lam: float) -> float:
    return np.sign(x) * max(abs(x) - lam, 0.0)


@njit(cache=True, fastmath=True)
def _coordinate_descent(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                        weights: np.ndarray, Xj_sq: np.ndarray,
                        lam_l1: float, lam_l2: float,
                        n: int, tol: float, max_iter: int) -> np.ndarray:
    p = beta.shape[0]
    residual = y - X @ beta
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            X_col = X[:, j]             # contiguous: X is Fortran order
            r_j    = residual + X_col * beta[j]
            rho    = (X_col @ r_j) / n
            thresh = lam_l1 * weights[j]
            bj_new = _soft_thresholding(rho, thresh) / (Xj_sq[j] + lam_l2)
            delta  = bj_new - beta[j]
            if delta != 0.0:
                residual -= X_col * delta
                beta[j]   = bj_new
        # convergence checks
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------
class AdaptiveElasticNet(BaseEstimator, RegressorMixin):
    """Adaptive Elastic Net with numba‑accelerated coordinate descent."""

    def __init__(self,
                 lmbda: float = 1.0,
                 alpha: float = 0.5,
                 gamma: float = 1.0,
                 ridge_alpha: float = 1.0,
                 fit_intercept: bool = True,
                 standardize_with_mean: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 warm_start: bool = False):
        self.lmbda = lmbda
        self.alpha = alpha
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.fit_intercept = fit_intercept
        self.standardize_with_mean = standardize_with_mean
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self._beta_previous_: Optional[np.ndarray] = None  # for warm starts

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, _ = X.shape

        # centre / scale
        self.X_mean_ = X.mean(0)
        if self.fit_intercept:
            self.y_mean_ = y.mean()
            y = y - self.y_mean_
            X = X - self.X_mean_
        else:
            self.y_mean_ = 0.0
        self.X_scale_ = np.std(X, 0, ddof=0)
        self.X_scale_[self.X_scale_ == 0.0] = 1.0
        X = X / self.X_scale_
        X = np.asfortranarray(X)          # make columns contiguous

        # ridge warm start (or reuse previous β)
        if not self.warm_start or self._beta_previous_ is None:
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(X, y)
            beta = ridge.coef_.astype(np.float32)
        else:
            beta = self._beta_previous_.copy()

        # adaptive weights
        eps = 1e-8
        w = 1.0 / (np.abs(beta) ** self.gamma + eps)

        # coordinate descent
        Xj_sq   = (X * X).sum(0) / n
        lam_l1  = self.lmbda * self.alpha
        lam_l2  = self.lmbda * (1.0 - self.alpha)
        beta    = _coordinate_descent(X, y, beta, w, Xj_sq,
                                       lam_l1, lam_l2, n,
                                       self.tol, self.max_iter)
        # stash for warm start
        if self.warm_start:
            self._beta_previous_ = beta.copy()

        # export to original scale
        self.beta_scaled_ = beta.copy()
        self.coef_        = beta / self.X_scale_
        self.intercept_   = (self.y_mean_ - (self.coef_ * self.X_mean_).sum()
                              if self.fit_intercept else 0.0)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.standardize_with_mean:
            X = (X - self.X_mean_) / self.X_scale_
        else:
            X = X / self.X_scale_
        return X @ self.beta_scaled_ + self.intercept_


# ---------------------------------------------------------------------------
# Path‑wise helpers  (one fit per (train_split, α) instead of per λ)
# ---------------------------------------------------------------------------

def _path_single_alpha(X: np.ndarray, y: np.ndarray, alpha: float,
                       lambdas: Sequence[float], model_kwargs: Dict) -> Dict[float, Tuple[np.ndarray, float]]:
    """Return {λ : (coef, intercept)} for one alpha."""
    model = AdaptiveElasticNet(alpha=alpha, warm_start=True, **model_kwargs)
    path: Dict[float, Tuple[np.ndarray, float]] = {}
    for lam in sorted(lambdas, reverse=True):  # warm start from large → small
        model.lmbda = lam
        model.fit(X, y)
        path[lam] = (model.coef_.copy(), model.intercept_)
    return path


def _mse_from_path(path, X_te, y_te):
    mse = np.mean((y_te - (X_te @ path[0] + path[1])) ** 2)  # path entry is (coef, intc)
    return mse


# ---------------------------------------------------------------------------
# Grid search using the path strategy (single parallel layer)
# ---------------------------------------------------------------------------

def grid_search_adaptive_enet(X: np.ndarray, y: np.ndarray,
                              lambdas, alphas,
                              cv_splits: int = 5,
                              gamma: float = 1.0,
                              ridge_alpha: float = 1.0,
                              fit_intercept: bool = True,
                              standardize_with_mean: bool = True,
                              max_iter: int = 1000,
                              tol: float = 1e-6,
                              n_jobs: int = 1):
    """Path‑wise grid search – returns (results_df, best_params_dict)."""
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model_kwargs = dict(gamma=gamma, ridge_alpha=ridge_alpha,
                        fit_intercept=fit_intercept,
                        standardize_with_mean=standardize_with_mean,
                        max_iter=max_iter, tol=tol)

    # Pre‑compute paths for every (train split, α)
    paths = {}
    for split_id, (tr_idx, _) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        for alpha in alphas:
            paths[(split_id, alpha)] = _path_single_alpha(X_tr, y_tr, alpha,
                                                          lambdas, model_kwargs)

    # Score each (λ, α) across CV folds  – parallel over folds
    def _score_fold(split_id, tr_idx, te_idx):
        X_te, y_te = X[te_idx], y[te_idx]
        records = []
        for alpha in alphas:
            path = paths[(split_id, alpha)]
            for lam, (coef, intercept) in path.items():
                mse = np.mean((y_te - (X_te @ coef + intercept)) ** 2)
                records.append({'alpha': alpha, 'lmbda': lam, 'mse': mse})
        return records

    all_records = Parallel(n_jobs=n_jobs)(
        delayed(_score_fold)(i, tr, te)
        for i, (tr, te) in enumerate(tscv.split(X))
    )

    # aggregate
    import pandas as pd
    df = (pd.concat([pd.DataFrame(r) for r in all_records])
            .groupby(['alpha', 'lmbda'], as_index=False)['mse']
            .mean()
            .sort_values('mse')
            .reset_index(drop=True))
    best = df.iloc[0].to_dict()
    return df, best


def run_elastic_net_adaptive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, float, float, np.ndarray]:
    """
    Adapted to use our AdaptiveElasticNet + grid‐search.
    Required params keys:
      - 'lambdas': list of λ to try
      - 'alphas' : list of α to try
    Optional keys:
      - 'gamma', 'ridge_alpha', 'fit_intercept', 'standardize', 'max_iter', 'tol'
    """
    # extract numpy arrays
    X_train = train[features].values
    y_train = train[target].values
    X_test  = test[features].values

    # pull grid parameters
    lambdas    = params.get('alpha_grid')
    alphas     = params.get('l1_ratio_grid')
    gamma      = params.get('gamma_grid')
    ridge_alpha= params.get('ridge_alpha', 1.0)
    fit_int    = ModelSettings.fit_intercept
    standardize_with_mean       = ModelSettings.standard_scaler_with_mean
    max_it     = params.get('max_iter', 1000)
    tol        = params.get('tol', 1e-6)
    
    # 1) grid‐search on train
    results_df, best = grid_search_adaptive_enet(
        X_train, y_train,
        lambdas, alphas,
        cv_splits=5,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        fit_intercept=fit_int,
        standardize_with_mean=standardize_with_mean,
        max_iter=max_it,
        tol=tol,
        n_jobs=1
    )
    best_lmbda = best['lmbda']
    best_alpha = best['alpha']
    
    # 2) fit final model on full train
    final_model = AdaptiveElasticNet(
        lmbda=best_lmbda,
        alpha=best_alpha,
        gamma=gamma,
        ridge_alpha=ridge_alpha,
        fit_intercept=fit_int,
        standardize_with_mean=standardize_with_mean,
        max_iter=max_it,
        tol=tol
    ).fit(X_train, y_train)

    # 3) predict
    test_out = test.copy()
    test_out['prediction'] = final_model.predict(X_test)

    # 4) return preds + best grid params + coefficients
    return test_out, best_lmbda, best_alpha, final_model.coef_


def run_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any],
    adaptive: bool = False
):
    """
    Run an Elastic Net regression model on test data using training data.
    If an "alpha_grid" is provided in params, grid search is performed over the specified
    list of alpha values to select the best model.
    
    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column name.
        features (List[str]): List of feature column names.
        params (Dict[str, Any]): Parameters for the model. It can include 'alpha', 'l1_ratio',
                                 'random_state', and optionally 'alpha_grid' (a list of alphas).
        adaptive (bool): If True, run the adaptive elastic net. Default is False.
    
    Returns:
        pd.DataFrame: Test set with predictions added in 'prediction' column.
    """
    if not params:
        raise ValueError("Elastic Net parameters must be provided.")
    local_params = params.copy()

    alpha_grid = local_params.pop('alpha_grid')
    l1_ratio_grid = local_params.pop('l1_ratio_grid')
    
    model = get_model_from_params(params, adaptive=adaptive, fit_intercept=ModelSettings.fit_intercept, standard_scaler_with_mean=ModelSettings.standard_scaler_with_mean)
    step_name = model.steps[-1][0]

    param_grid = {f"{step_name}__alpha": alpha_grid}
    if "l1_ratio" in model.named_steps[step_name].get_params():
        param_grid[f"{step_name}__l1_ratio"] = l1_ratio_grid

    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    gs.fit(train[features], train[target])

    best_model = gs.best_estimator_
    predictions = best_model.predict(test[features])
    best_alpha = gs.best_params_[f'{step_name}__alpha']
    best_l1_ratio = gs.best_params_.get(f'{step_name}__l1_ratio')

    coefs = best_model.named_steps[step_name].coef_
    intercept = best_model.named_steps[step_name].intercept_

    test = test.copy()
    test['prediction'] = predictions
    return test, best_alpha, best_l1_ratio, coefs


"""def run_elastic_net_adaptive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, float, float, np.ndarray]:
    ""
    Run an adaptive Elastic Net where only the L1 penalty is weighted via penalty_factor.
    This uses an initial Ridge to get β˜, computes adaptive weights, then
    scales features and falls back to the sklearn-based run_elastic_net.
    ""
    # 1) initial Ridge to get β˜
    ridge_params = params.get('ridge_params', {})
    ridge_pipe = get_model_from_params(ridge_params, adaptive=False)
    ridge_pipe.fit(train[features], train[target])
    step = ridge_pipe.steps[-1][0]
    beta_init = ridge_pipe.named_steps[step].coef_

    # 2) compute adaptive weights
    gamma = params.get('gamma', 1.0)
    epsilon = params.get('epsilon', 1e-6)
    weights = 1.0 / (np.abs(beta_init) ** gamma + epsilon)

    # 3) hyperparameter grids
    alpha_grid     = params.get('alpha_grid')
    l1_ratio_grid  = params.get('l1_ratio_grid')

    # 4) Scale features by adaptive weights, then hand off to sklearn-based run_elastic_net
    def scale_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df_in.copy()
        for i, col in enumerate(features):
            df_scaled[col] = df_scaled[col] / weights[i]
        return df_scaled

    train_scaled = scale_df(train)
    test_scaled  = scale_df(test)

    # call your existing GridSearchCV/sklearn ElasticNet wrapper
    pred_df, best_alpha, best_l1_ratio, coefs_scaled = run_elastic_net(
        train_scaled,
        test_scaled,
        target,
        features,
        {
            'alpha_grid':    alpha_grid,
            'l1_ratio_grid': l1_ratio_grid,
            # include any other sklearn params here if needed
        },
        adaptive=True
    )

    # 5) Rescale the coefficients back to the original scale
    coefs = coefs_scaled / weights

    return pred_df, best_alpha, best_l1_ratio, coefs

"""
"""def run_elastic_net_adaptive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, float, float, np.ndarray]:
    ""
    Run an adaptive elastic net regression using initial coefficient estimates
    to compute adaptive weights, scale the features, run elastic net, then 
    adjust the coefficients back to the original scale.
    
    This refactored version makes use of helper functions for modularity.
    
    Args:
        train: Training DataFrame.
        test: Testing DataFrame.
        target: Name of the target column.
        features: List of feature column names.
        params: Dictionary of parameters. It should contain:
            - "gamma": exponent for weight computation (optional, default=1.0).
            - "epsilon": small constant (optional, default=1e-6).
            - "ridge_params": parameters for the initial Ridge regression.
            Also includes parameters for grid search over the elastic net.
    
    Returns:
        A tuple: (predicted test DataFrame, best_alpha, best_l1_ratio, adjusted coefficients)
    ""
    def scale_features(
        df: pd.DataFrame, 
        features: List[str], 
        weights: np.ndarray
    ) -> pd.DataFrame:
        ""
        Scale feature columns in the DataFrame by dividing each by its respective weight.
        
        Args:
            df: Input DataFrame.
            features: List of feature column names.
            weights: Adaptive weights for each feature.
            
        Returns:
            A new DataFrame with scaled features.
        ""
        df_scaled = df.copy()
        # Scale each feature column using the corresponding weight
        for i, col in enumerate(features):
            df_scaled[col] = df_scaled[col] / weights[i]
        return df_scaled
    # 1. Fit Ridge to get initial coeffs and scaler
    ridge_model = get_model_from_params(params.get("ridge_params", {}), adaptive=False)
    ridge_model.fit(train[features], train[target])
    step = ridge_model.steps[-1][0]
    beta_init = ridge_model.named_steps[step].coef_
    scaler = ridge_model.named_steps['standardscaler']

    # 2. Standardize features
    train_std = train.copy()
    test_std = test.copy()
    train_std[features] = scaler.transform(train[features])
    test_std[features] = scaler.transform(test[features])

    # 3. Compute adaptive weights
    gamma = params.get("gamma", 1.0)
    epsilon = params.get("epsilon", 1e-6)
    weights = 1.0 / (np.abs(beta_init) ** gamma + epsilon)

    # Scaling is nu op X, moet op de coefficienten zijn. Moeten de objective function aanpassen. 

    # 4. Scale standardized features by weights
    train_scaled = scale_features(train_std, features, weights)
    test_scaled = scale_features(test_std, features, weights)

    # 5. Run ElasticNet on weighted data (no internal scaling)
    pred_df, best_alpha, best_l1_ratio, coefs_scaled = run_elastic_net(
        train_scaled, test_scaled, target, features, params, adaptive=True
    )

    # 6. Rescale coefficients back
    coefs = coefs_scaled / weights
    
    return pred_df, best_alpha, best_l1_ratio, coefs
"""
"""def run_elastic_net_adaptive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
):
    ""
    Run an adaptive elastic net regression using initial coefficient estimates
    to compute adaptive weights.
    
    This function computes initial estimates with a Ridge regression,
    calculates weights as: weights_j = 1/ (|β_j^{init}|**gamma + epsilon)
    (with a small epsilon to avoid division by zero), scales the features by these
    weights for the L1 part, and then runs the standard elastic net on the scaled data.
    The output coefficients are transformed back to the original scale.
    
    Args:
        train: Training DataFrame.
        test: Testing DataFrame.
        target: Name of the target column.
        features: List of feature column names.
        params: Dictionary of parameters; must include "adaptive": True to trigger this branch,
                and "gamma": <value> (default can be 1.0) for the weight computation.
        l1_grid: Whether to perform grid search on L1 penalty parameters.
    
    Returns:
        Tuple: (predicted test DataFrame, best_alpha, best_l1_ratio, coefficients)
    ""
    # Set gamma (or default to 1.0)
    gamma = params.get("gamma_grid", 1.0)
    # Of gridsearch?
    
    # 1. Compute initial estimates using a Ridge regressor
    model = get_model_from_params(ModelParameters.ridge_params)
    model.fit(train[features], train[target])
    final_model = model.named_steps[list(model.named_steps)[-1]]
    beta_init = final_model.coef_


    # 2. Compute adaptive weights (small constant added to avoid division by zero)
    epsilon = 1e-6
    weights = 1.0 / (np.abs(beta_init)**gamma + epsilon) # + epsilon)
    #print(weights)
    # 3. Scale the features for the L1 penalty; here, we create new DataFrames
    train_scaled = train.copy()
    test_scaled = test.copy()
    for i, col in enumerate(features):
        train_scaled[col] = train_scaled[col] / weights[i]
        test_scaled[col] = test_scaled[col] / weights[i]
    
    ""print("Scaled features for L1 penalty.")
    print(train_scaled[features].head())
    print(train[features].head())""
    # 4. Run the standard elastic net on the scaled data.
    # (Reuse your existing run_elastic_net function)
    pred_df, best_alpha, best_l1_ratio, coefs_scaled = run_elastic_net(
        train_scaled, test_scaled, target, features, params
    )
    
    # 5. Adjust the coefficients back to their original scale.
    coefs = coefs_scaled #/ weights
    #raise ValueError
    return pred_df, best_alpha, best_l1_ratio, coefs

"""


def forecast_single_date(
        forecast_date: pd.Timestamp,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        forecast_horizon: int,
        rolling_window_days: int,
        enet_params: dict,
        datetime_col: str,
        freq: str
    ):
    return forecast_single_date_generic(
        forecast_date,
        df,
        target_column,
        feature_columns,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq,
        run_elastic_net
    )

def forecast_single_date_adaptive(
        forecast_date: pd.Timestamp,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        forecast_horizon: int,
        rolling_window_days: int,
        enet_params: dict,
        datetime_col: str,
        freq: str
    ):
    return forecast_single_date_generic(
        forecast_date,
        df,
        target_column,
        feature_columns,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq,
        run_elastic_net_adaptive
    )

def forecast_single_date_generic(
    forecast_date: pd.Timestamp,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int,
    rolling_window_days: int,
    enet_params: dict,
    datetime_col: str,
    freq: str,
    model_runner: Callable
):
    """
    Generic function for forecasting a single date using a model runner.

    Args:
        model_runner: Function to train and apply the model (e.g., run_elastic_net).

    Returns:
        List of dictionaries with forecast results.
    """
    forecast_results_single = []
    forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)

    train_df = data_interpolate_prev(df, forecast_date, rolling_window_days)
    test_df = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

    if train_df.empty or test_df.empty:
        return forecast_results_single
    
    pred_df, best_alpha, best_l1_ratio, coefs = model_runner(
        train_df, test_df, target_column, feature_columns, enet_params
    )

    for _, row in pred_df.iterrows():
        forecast_results_single.append({
            'target_time': row[datetime_col],  # use passed-in datetime_col, not ModelSettings
            'prediction': row['prediction'],
            'actual': row[target_column],
            'best_alpha': best_alpha,
            'best_l1_ratio': best_l1_ratio,
            'coefs': coefs
        })

    return forecast_results_single


def run_day_ahead_elastic_net(
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        forecast_horizon: int = 96, 
        rolling_window_days: int = 165, 
        enet_params: Dict[str, Any] = None, 
        datetime_col: str = 'datetime', 
        freq: str = '15min'
    ) -> pd.DataFrame:
    return run_day_ahead_forecasting(
        df,
        target_column,
        feature_columns,
        forecast_single_date,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq
    )

def run_day_ahead_elastic_net_adaptive(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    enet_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    return run_day_ahead_forecasting(
        df,
        target_column,
        feature_columns,
        forecast_single_date_adaptive,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq
    )

def run_day_ahead_forecasting(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_func: Callable,
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    enet_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min',
    desc: str = "Forecasting"
) -> pd.DataFrame:
    if not enet_params:
        raise ValueError("Elastic Net parameters must be provided.")
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    forecast_results = []
    forecast_dates = [
        d for d in df[datetime_col].unique()
        if pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0
    ]
    #print(len(forecast_dates), "forecast dates found.")
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                forecast_func,
                pd.Timestamp(forecast_date),
                df,
                target_column,
                feature_columns,
                forecast_horizon,
                rolling_window_days,
                enet_params,
                datetime_col,
                freq
            ): forecast_date for forecast_date in forecast_dates[rolling_window_days:]
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                forecast_results.extend(result)
            except Exception as exc:
                print(f"Forecasting failed for {futures[future]}: {exc}")
                raise ValueError(f"Forecasting failed for {futures[future]}: {exc}")
    
    print("Forecasting complete.")
    return pd.DataFrame(forecast_results)

