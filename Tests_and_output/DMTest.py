import numpy as np
import pandas as pd
from scipy import stats


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

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Path to your CSV file
    csv_path = "/Users/borisrine/Library/CloudStorage/OneDrive-ErasmusUniversityRotterdam/Documents/Uni/bsc2/Year 4/Seminar/Python tests/combined_forecast321.csv"
    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    forecast_cols = [col for col in df.columns if col.startswith("A")]
    print("Forecast models found:", forecast_cols)

    if "HR" not in df.columns:
        raise ValueError("Column 'HR' with actual values not found.")

    df_clean = df[["HR"] + forecast_cols].dropna()

    # Create empty string matrix for results
    results_matrix = pd.DataFrame("", index=forecast_cols, columns=forecast_cols)

    for i in range(len(forecast_cols)):
        for j in range(i + 1, len(forecast_cols)):
            model_i = forecast_cols[i]
            model_j = forecast_cols[j]

            try:
                result = DMTest.run(
                    forecast1=df_clean[model_i],
                    forecast2=df_clean[model_j],
                    actual=df_clean["HR"],
                    alternative="two-sided"
                )
                dm_stat = result["dm_stat"]
                p_value = result["p_value"]

                # Format: stat (pval)
                entry = f"{dm_stat:.2f} ({p_value:.3f})"
                results_matrix.loc[model_i, model_j] = entry
                results_matrix.loc[model_j, model_i] = entry

            except Exception as e:
                print(f"Error comparing {model_i} vs {model_j}: {e}")
                results_matrix.loc[model_i, model_j] = "ERR"
                results_matrix.loc[model_j, model_i] = "ERR"

    # Fill diagonal with dashes
    np.fill_diagonal(results_matrix.values, "—")

    print("\n📊 Diebold-Mariano Test Statistic Matrix (statistic with p-value):")
    print(results_matrix)