import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt


class DMTest:
    @staticmethod
    def run(forecast1: pd.Series,
            forecast2: pd.Series,
            actual: pd.Series,
            alternative: str = "two-sided",
            lag: int = None) -> dict:
        """
        Computes the Diebold-Mariano test statistic for two forecast error series.

        Parameters:
        - forecast1, forecast2: Forecast values (pd.Series)
        - actual: Actual values (pd.Series)        
        - alternative: str, one of {'two-sided', 'greater', 'less'}
        - lag: int, optional, lag order for HAC variance estimate (default: floor(T^(1/3)))

        Returns:
        - dict with 'dm_stat', 'p_value', and 'standard_error'
        """
        if not isinstance(forecast1, pd.Series) or not isinstance(forecast2, pd.Series) or not isinstance(actual, pd.Series):
            raise ValueError("Inputs must be pandas Series")

        if not (len(forecast1) == len(forecast2) == len(actual)):
            raise ValueError("Forecast error series must be of the same length")

        errors1 = actual - forecast1
        errors2 = actual - forecast2
        # Calculate loss differentials (squared errors)
        d = (errors1**2 - errors2**2).dropna()
        T = len(d)

        if T < 5:
            raise ValueError("Too few observations for DM test")

        # Set default lag based on DM paper recommendation if not specified
        if lag is None:
            lag = int(np.floor(T ** (1/3)))

        # Compute mean differential and long-run variance using Newey-West
        mean_d = np.mean(d)

        # Newey-West estimator for variance
        gamma_0 = np.var(d, ddof=0)
        gamma = [np.cov(d[:-k], d[k:], ddof=0)[0, 1] for k in range(1, lag + 1)]
        long_run_var = gamma_0 + 2 * np.sum([ (1 - k/(lag+1)) * g for k, g in enumerate(gamma, start=1)])

        # Compute DM statistic
        dm_stat = mean_d / np.sqrt(long_run_var / T)

        # Compute p-value
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(dm_stat)
        elif alternative == "less":
            p_value = stats.norm.cdf(dm_stat)
        else:
            raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'greater', 'less'.")

        standard_error = np.sqrt(long_run_var / T)

        return {
            "dm_stat": dm_stat,
            "p_value": p_value,
            "standard_error": standard_error
        }


def check_forecasts(file1: str, file2: str, prediction_name: str = 'prediction', actual_name: str = 'prediction') -> None:
    """
    Reads two CSV files containing forecasts and actuals, checks their compatibility,
    and runs the Diebold-Mariano test.

    Parameters:
    - file1: Path to the first CSV file (e.g., forecasts from DLasso)
    - file2: Path to the second CSV file (e.g., forecasts from Delnet)
    - prediction_name: The name of the prediction column (default: "y_pred")
    - actual_name: The name of the actual column (default: "y_actual")
    """
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1.sort_values(by="target_time", inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.sort_values(by="target_time", inplace=True)
    df2.reset_index(drop=True, inplace=True)
    
    # Check if the necessary columns exist in each file using the provided column names
    for df, path in zip([df1, df2], [file1, file2]):
        if prediction_name not in df.columns or actual_name not in df.columns:
            raise ValueError(f"File '{path}' must contain both '{prediction_name}' and '{actual_name}' columns.")

    # Extract forecast and actual series based on the dynamic column names
    forecast1 = df1[prediction_name]
    actual1 = df1[actual_name]
    forecast2 = df2[prediction_name]
    actual2 = df2[actual_name]

    # Check that both actual series are equivalent; warn if not
    if not np.array_equal(actual1.to_numpy(), actual2.to_numpy()):
        raise ValueError("Warning: Actual values differ between the two files. Using the actuals from the first file.")
   
        
    # Use the actual series from the first file
    actual = actual1

    # Combine the series into one DataFrame (to drop any rows with missing values)
    combined = pd.DataFrame({
        "forecast1": forecast1,
        "forecast2": forecast2,
        "actual": actual
    }).dropna()

    # Run the DM Test using the combined data
    result = DMTest.run(
        forecast1=combined["forecast1"],
        forecast2=combined["forecast2"],
        actual=combined["actual"],
        alternative="two-sided",
        lag=96
    )

    # Print the test results
    print("Diebold-Mariano Test Results:")
    print(f"DM Statistic: {result['dm_stat']:.2f}")
    print(f"p-value: {result['p_value']:.3f}")
    print(f"Standard Error: {result['standard_error']:.4f}")

def check_all_forecasts_solar_data(
    forecast_files: dict,
    prediction_name: str = "prediction",
    actual_name: str = "actual",
    time_col: str = "target_time",
) -> pd.DataFrame:
    """
    Run pair‑wise Diebold–Mariano tests on every combination of
    forecast files.

    Parameters
    ----------
    forecast_files : dict
        {model_name: path_to_csv}.  Each CSV must contain columns
        `prediction_name`, `actual_name`, and `time_col`.
    prediction_name, actual_name : str
        Column names for the point forecast and the realised value.
    time_col : str
        Timestamp column (parsed with ``pd.to_datetime``).

    Returns
    -------
    pd.DataFrame
        Columns: model_1, model_2, dm_stat, p_value, standard_error
    """
    # ------------------------------------------------------------------
    # 1. read every file once, tidy, and keep in a dict
    # ------------------------------------------------------------------
    def _load(path):
        df = pd.read_csv(path)

        missing = {c for c in (prediction_name, actual_name, time_col)
                   if c not in df.columns}
        if missing:
            raise ValueError(f"{path} is missing column(s): {missing}")

        return (
            df.assign(**{time_col: pd.to_datetime(df[time_col])})
              .set_index(time_col)
              .sort_index()
              [[prediction_name, actual_name]]
              .rename(columns={prediction_name: "pred", actual_name: "act"})
        )

    frames = {name: _load(p) for name, p in forecast_files.items()}
    results = []

    # ------------------------------------------------------------------
    # 2. pair‑wise DM tests
    # ------------------------------------------------------------------
    for m1, m2 in combinations(frames, 2):
        joined = (
            frames[m1].rename(columns={"pred": "pred1"})
            .join(frames[m2]["pred"].rename("pred2"), how="inner")
            .dropna()                     # drop rows with any NaN
        )

        if joined.empty:
            raise ValueError(f"No common timestamps for {m1} vs {m2}")

        # Run Diebold–Mariano on aligned series
        dm = DMTest.run(
            forecast1=joined["pred1"],
            forecast2=joined["pred2"],
            actual   =joined["act"],      # same 'act' column for both
            alternative="two-sided",
            lag = 96
        )

        results.append(
            dict(model_1=m1, model_2=m2,
                 dm_stat=dm["dm_stat"],
                 p_value=dm["p_value"],
                 standard_error=dm["standard_error"])
        )

    return pd.DataFrame(results)

def dm_diagnostics(
    forecast1: pd.Series,
    forecast2: pd.Series,
    actual:   pd.Series,
    alt: str = "two-sided",
    obs_per_day: int = 96,
    nw_lag: int | None = None,
    plot: bool = True,
):
    """
    One‑stop health‑check for a Diebold‑Mariano comparison.
    Returns DM‑stat, p‑value, HAC s.e., long‑run variance … and
    (optionally) shows ACF + cumulative‑sum plots of d_t.
    """
    if not (isinstance(forecast1, pd.Series) and
            isinstance(forecast2, pd.Series) and
            isinstance(actual,   pd.Series)):
        raise TypeError("Inputs must be pandas Series.")
    if not (len(forecast1) == len(forecast2) == len(actual)):
        raise ValueError("Series must have equal length.")

    d = ((actual - forecast1)**2 - (actual - forecast2)**2).dropna()
    T = len(d)
    d_mean = d.mean()

    # ---------- diagnostic plots ----------
    if plot:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout="tight")
        plot_acf(d - d_mean, lags=min(10*obs_per_day, T//2),
                 zero=False, ax=ax[0])
        ax[0].set_title("ACF of loss differentials (d_t)")
        (d - d_mean).cumsum().plot(ax=ax[1])
        ax[1].axhline(0, ls="--", c="k")
        ax[1].set_title("Cumulative sum of centred d_t")
        plt.show()

    # ---------- Newey–West long‑run variance ----------
    L = obs_per_day if nw_lag is None else nw_lag
    gamma0 = ((d - d_mean)**2).mean()
    gamma = [((d[k:] - d_mean)*(d[:-k] - d_mean)).mean()
             for k in range(1, L+1)]
    long_run_var = (gamma0 +
                    2*sum((1 - k/(L+1))*g for k, g in enumerate(gamma, 1)))
    se = np.sqrt(long_run_var / T)
    dm_stat = d_mean / se

    if   alt == "two-sided": p = 2*(1 - stats.norm.cdf(abs(dm_stat)))
    elif alt == "greater" : p = 1 - stats.norm.cdf(dm_stat)
    elif alt == "less"    : p =     stats.norm.cdf(dm_stat)
    else: raise ValueError("alt must be two-sided/greater/less")

    return dict(dm_stat=dm_stat, p_value=p, standard_error=se,
                long_run_var=long_run_var, bandwidth_used=L,
                mean_loss_diff=d_mean, n_obs=T)

# ----------------------------------------------------------------------
#  SMALL RUNNER that plugs the helper into your CSVs -------------------
# ----------------------------------------------------------------------
def dm_diag_from_csv(path1: str, path2: str,
                     prediction_name="prediction", actual_name="actual",
                     time_col="target_time",
                     obs_per_day=96, nw_lag=None, plot=True):
    """
    Loads two forecast CSVs + realised series (same I/O logic as your
    _load() function), aligns them on timestamps, then feeds them to
    dm_diagnostics().
    """
    def _load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        missing = {c for c in (prediction_name, actual_name, time_col)
                if c not in df.columns}
        if missing:
            raise ValueError(f"{path} is missing column(s): {missing}")

        # make timestamp a proper index
        df[time_col] = pd.to_datetime(df[time_col])
        df = (df.set_index(time_col)
                .sort_index()
                [[prediction_name, actual_name]]
                .copy())

        return df

    df1, df2 = map(_load, (path1, path2))
    joined = (df1.rename(columns={prediction_name: "pred1",
                                  actual_name: "act"})
                 .join(df2[prediction_name].rename("pred2"),
                       how="inner")
                 .dropna())
    if joined.empty:
        raise ValueError("No overlapping timestamps between the files.")

    res = dm_diagnostics(joined["pred1"], joined["pred2"], joined["act"],
                         obs_per_day=obs_per_day, nw_lag=nw_lag, plot=plot)
    print("\nDM‑diagnostics")
    for k, v in res.items():
        print(f"{k:16s}: {v}")
    return res



def main(forecast_files): 

    order = list(forecast_files)        # <- preserve that exact sequence
    # –– 2  run DM tests –––––––––––––––––––––––––––––––––––––––––––––––– #
    dm_long = check_all_forecasts_solar_data(forecast_files)
    dm_long.to_csv("dm_pairwise_results.csv", index=False)

    # build “stat (p)” text for each pair
    dm_long["txt"] = (dm_long["dm_stat"].round(2).astype(str)
                      + " ("+dm_long["p_value"].round(3).astype(str)+")")

    matrix = dm_long.pivot(index="model_1", columns="model_2", values="txt")
    matrix = matrix.combine_first(matrix.T)      # mirror upper→lower
    for m in order:                              # blank diagonal
        matrix.loc[m, m] = ""

    # –– 3  enforce desired row/column order –––––––––––––––––––––––––––– #
    matrix = matrix.reindex(index=order, columns=order)

    # –– 4  nice terminal print ––––––––––––––––––––––––––––––––––––––––– #
    try:
        from tabulate import tabulate
        print("\nDiebold–Mariano  stat (p‑val)\n")
        print(tabulate(matrix.fillna(""), headers="keys",
                       showindex=True, tablefmt="fancy_grid"))
    except ImportError:
        print(matrix.fillna(""))

    # –– 5  write to Excel ––––––––––––––––––––––––––––––––––––––––––––– #
    with pd.ExcelWriter("dm_matrix_stat_and_p.xlsx") as xl:
        matrix.to_excel(xl, sheet_name="DM_stat_(p)")

    print("\n✔  Matrix saved to 'dm_matrix_stat_and_p.xlsx' in dictionary order")
    

def dm_diagnostics(
    forecast1: pd.Series,
    forecast2: pd.Series,
    actual:   pd.Series,
    alt: str = "two-sided",
    obs_per_day: int = 96,      # 96 for 15‑min data, 24 for hourly, …
    nw_lag: int | None = None,  # override if you really want to
    plot: bool = True,
):
    """
    Quick health‑check for Diebold–Mariano inputs.

    1.  computes the loss‑differential series d_t = e1_t² - e2_t²
    2.  shows its ACF             (optional)
    3.  recommends an NW bandwidth (= obs_per_day)
    4.  returns DM‑stat, p‑value, SE, and long‑run variance

    Parameters
    ----------
    forecast1, forecast2, actual : pd.Series aligned on the same index
    alt   : "two-sided" | "greater" | "less"
    obs_per_day : #observations per diurnal cycle (96 = 15‑min)
    nw_lag : if None  →  obs_per_day is used
    plot  : if True   →  ACF and cumulative sum charts are shown
    """
    if not (isinstance(forecast1, pd.Series) and
            isinstance(forecast2, pd.Series) and
            isinstance(actual,   pd.Series)):
        raise TypeError("Inputs must be pandas Series.")

    if not (len(forecast1) == len(forecast2) == len(actual)):
        raise ValueError("Series must have equal length.")

    # ------------------------------------------------------------------
    # 1. construct loss differential -----------------------------------
    # ------------------------------------------------------------------
    d = ( (actual - forecast1)**2 - (actual - forecast2)**2 ).dropna()
    T = len(d)
    if T < 20:
        raise ValueError("Too few observations (<20) for diagnostics.")

    d_mean = d.mean()

    # ------------------------------------------------------------------
    # 2. ACF & cum‑sum plots ------------------------------------------
    # ------------------------------------------------------------------
    if plot:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout="tight")

        plot_acf(d - d_mean, lags=min(10*obs_per_day, T//2),
                 zero=False, ax=ax[0])
        ax[0].set_title("ACF of loss differentials")

        (d - d_mean).cumsum().plot(ax=ax[1])
        ax[1].axhline(0, ls="--", c="k")
        ax[1].set_title("Cumulative sum of centred dₜ")
        plt.show()

    # ------------------------------------------------------------------
    # 3. Newey–West long‑run variance ---------------------------------
    # ------------------------------------------------------------------
    L = obs_per_day if nw_lag is None else nw_lag
    gamma0 = ((d - d_mean)**2).mean()

    gamma = [ ((d[k:] - d_mean)*(d[:-k] - d_mean)).mean()
              for k in range(1, L+1) ]

    long_run_var = (gamma0 +
                    2*sum((1 - k/(L+1))*g for k, g in enumerate(gamma, 1)))

    se = np.sqrt(long_run_var / T)
    dm_stat = d_mean / se

    if alt == "two-sided":
        p = 2*(1 - stats.norm.cdf(abs(dm_stat)))
    elif alt == "greater":
        p = 1 - stats.norm.cdf(dm_stat)
    elif alt == "less":
        p = stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alt must be 'two-sided', 'greater', or 'less'.")

    return {
        "dm_stat": dm_stat,
        "p_value": p,
        "standard_error": se,
        "long_run_var": long_run_var,
        "bandwidth_used": L,
        "mean_loss_diff": d_mean,
        "n_obs": T,
    }



def main_price(forecast_files): 

    order = list(forecast_files)        # <- preserve that exact sequence
    # –– 2  run DM tests –––––––––––––––––––––––––––––––––––––––––––––––– #
    dm_long = check_all_forecasts_solar_data(forecast_files)
    dm_long.to_csv("dm_pairwise_results.csv", index=False)

    # build “stat (p)” text for each pair
    dm_long["txt"] = (dm_long["dm_stat"].round(2).astype(str)
                      + " ("+dm_long["p_value"].round(3).astype(str)+")")

    matrix = dm_long.pivot(index="model_1", columns="model_2", values="txt")
    matrix = matrix.combine_first(matrix.T)      # mirror upper→lower
    for m in order:                              # blank diagonal
        matrix.loc[m, m] = ""

    # –– 3  enforce desired row/column order –––––––––––––––––––––––––––– #
    matrix = matrix.reindex(index=order, columns=order)

    # –– 4  nice terminal print ––––––––––––––––––––––––––––––––––––––––– #
    try:
        from tabulate import tabulate
        print("\nDiebold–Mariano  stat (p‑val)\n")
        print(tabulate(matrix.fillna(""), headers="keys",
                       showindex=True, tablefmt="fancy_grid"))
    except ImportError:
        print(matrix.fillna(""))

    # –– 5  write to Excel ––––––––––––––––––––––––––––––––––––––––––––– #
    with pd.ExcelWriter("dm_matrix_stat_and_p_pricedata.xlsx") as xl:
        matrix.to_excel(xl, sheet_name="DM_stat_(p)")

    print("\n✔  Matrix saved to 'dm_matrix_stat_and_p.xlsx' in dictionary order")
    

def dm_diagnostics(
    forecast1: pd.Series,
    forecast2: pd.Series,
    actual:   pd.Series,
    alt: str = "two-sided",
    obs_per_day: int = 96,      # 96 for 15‑min data, 24 for hourly, …
    nw_lag: int | None = None,  # override if you really want to
    plot: bool = True,
):
    """
    Quick health‑check for Diebold–Mariano inputs.

    1.  computes the loss‑differential series d_t = e1_t² - e2_t²
    2.  shows its ACF             (optional)
    3.  recommends an NW bandwidth (= obs_per_day)
    4.  returns DM‑stat, p‑value, SE, and long‑run variance

    Parameters
    ----------
    forecast1, forecast2, actual : pd.Series aligned on the same index
    alt   : "two-sided" | "greater" | "less"
    obs_per_day : #observations per diurnal cycle (96 = 15‑min)
    nw_lag : if None  →  obs_per_day is used
    plot  : if True   →  ACF and cumulative sum charts are shown
    """
    if not (isinstance(forecast1, pd.Series) and
            isinstance(forecast2, pd.Series) and
            isinstance(actual,   pd.Series)):
        raise TypeError("Inputs must be pandas Series.")

    if not (len(forecast1) == len(forecast2) == len(actual)):
        raise ValueError("Series must have equal length.")

    # ------------------------------------------------------------------
    # 1. construct loss differential -----------------------------------
    # ------------------------------------------------------------------
    d = ( (actual - forecast1)**2 - (actual - forecast2)**2 ).dropna()
    T = len(d)
    if T < 20:
        raise ValueError("Too few observations (<20) for diagnostics.")

    d_mean = d.mean()

    # ------------------------------------------------------------------
    # 2. ACF & cum‑sum plots ------------------------------------------
    # ------------------------------------------------------------------
    if plot:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout="tight")

        plot_acf(d - d_mean, lags=min(10*obs_per_day, T//2),
                 zero=False, ax=ax[0])
        ax[0].set_title("ACF of loss differentials")

        (d - d_mean).cumsum().plot(ax=ax[1])
        ax[1].axhline(0, ls="--", c="k")
        ax[1].set_title("Cumulative sum of centred dₜ")
        plt.show()

    # ------------------------------------------------------------------
    # 3. Newey–West long‑run variance ---------------------------------
    # ------------------------------------------------------------------
    L = obs_per_day if nw_lag is None else nw_lag
    gamma0 = ((d - d_mean)**2).mean()

    gamma = [ ((d[k:] - d_mean)*(d[:-k] - d_mean)).mean()
              for k in range(1, L+1) ]

    long_run_var = (gamma0 +
                    2*sum((1 - k/(L+1))*g for k, g in enumerate(gamma, 1)))

    se = np.sqrt(long_run_var / T)
    dm_stat = d_mean / se

    if alt == "two-sided":
        p = 2*(1 - stats.norm.cdf(abs(dm_stat)))
    elif alt == "greater":
        p = 1 - stats.norm.cdf(dm_stat)
    elif alt == "less":
        p = stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alt must be 'two-sided', 'greater', or 'less'.")

    return {
        "dm_stat": dm_stat,
        "p_value": p,
        "standard_error": se,
        "long_run_var": long_run_var,
        "bandwidth_used": L,
        "mean_loss_diff": d_mean,
        "n_obs": T,
    }

if __name__ == "__main__":
    forecast_files = {
       "Simple_average": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/filtered_simple_average_forecast.csv',
        "lasso": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/lasso_forecast (3).csv",
        "ridge": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/ridge_forecast (3).csv",
        "elastic_net_fixed": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/elastic_net_forecast (3).csv",
        "tuned_elastic_net": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/tune_elastic_net_forecast.csv",
       "Adaptive Elastic net": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/adaptive_elastic_net_forecast (1).csv',
       "Weighted Window": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/WEWResults.csv",
        "Bayesian": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/filtered_bayesian_forecast.csv',
        "XGBoost": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/xgboost_forecast.csv',
        'Transformer': '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_solar/transformer_forecast_results_with_mean_True.csv'
    }

      
    forecast_files_price = {
        "Simple_average": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/simple_average_forecast_price.csv',
        "lasso": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/lasso_forecast_price.csv",
        "ridge": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/ridge_forecast_price.csv",
        "elastic_net_fixed": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/elastic_net_forecast_price.csv",
        "tuned_elastic_net": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/tune_elastic_net_forecast_price.csv",
        "adaptive elastic net": "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/adaptive_elastic_net_forecast_price.csv",
        "Bayesian": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/ForecastBayesian_price.csv',
        "XGBoost": '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/xgboost_forecast_price.csv',
        'Transfomer': '/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Data simuleren/DMtest/results_price/transformer_forecast_results_price.csv'
    }
    main(forecast_files)
    #main_price(forecast_files_price)