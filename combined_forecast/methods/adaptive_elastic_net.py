"""
Adaptive Elastic Net Regression using the shared Elastic Net model pipeline
"""
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from combined_forecast.methods.elastic_net import get_model_from_params


def run_adaptive_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Runs adaptive Elastic Net regression with GridSearchCV using shared pipeline logic.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): Target column name.
        features (List[str]): Feature column names.
        params (Dict[str, Any]): Parameters for search and model.

    Returns:
        pd.DataFrame: Test data with 'prediction' column.
    """
    # GridSearch-related defaults
    param_grid = params.get('param_grid', {
        'elasticnet__alpha': [0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    })

    grid_params = {
        'cv': params.get('cv', 5),
        'n_jobs': params.get('n_jobs', -1),
        'verbose': params.get('verbose', 0)
    }

    # Extract base model pipeline and model step
    base_params = {
        'alpha': params.get('alpha', 1.0),
        'l1_ratio': params.get('l1_ratio', 0.5),
        'random_state': params.get('random_state', 42)
    }

    pipeline = get_model_from_params(base_params)
    model_name = pipeline.steps[-1][0]  # usually 'elasticnet' or 'ridge'

    # Rename model step in pipeline for GridSearchCV to work
    pipeline.steps[-1] = (model_name, pipeline.steps[-1][1])

    grid = GridSearchCV(pipeline, param_grid=param_grid, **grid_params)
    grid.fit(train[features], train[target])

    test = test.copy()
    test['prediction'] = grid.predict(test[features])
    test['actual'] = test[target] 
    return test



if __name__ == '__main__':
    from combined_forecast.utils import generate_sample_data, evaluate_forecast

    # Generate sample 15-min interval data for 20 days
    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    # Split last day as test, rest as training
    train_df = df_sample.iloc[:-96]
    test_df = df_sample.iloc[-96:]

    # Run Adaptive Elastic Net with grid search
    forecast_df = run_adaptive_elastic_net(
        train=train_df,
        test=test_df,
        target=target_col,
        features=feature_cols,
        params={
            'param_grid': {
                'elasticnet__alpha': [0.1, 1.0, 10.0],
                'elasticnet__l1_ratio': [0.0, 0.5, 1.0]
            },
            'cv': 5,
            'n_jobs': -1,
            'verbose': 1
        }
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\n RMSE on Adaptive Elastic Net forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")
