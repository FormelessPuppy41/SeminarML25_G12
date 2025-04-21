import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

def load_forecast_data(file_path):
    """
    Loads wind forecast data from the given CSV file and returns a DataFrame
    with selected columns.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with selected forecast and metadata columns.
    """
    # Read the full CSV file
    df = pd.read_csv(file_path)

    # Select relevant columns
    selected_columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "K", "HR", "Zeit", "Jahr", "Monat", "MW_J"]
    df_selected = df[selected_columns].copy()

    return df_selected


def load_parameters(file_path):
    """
    Loads parameters from a CSV file into a dictionary.

    Parameters:
        file_path (str): Path to the parameter CSV file.

    Returns:
        dict: Dictionary of parameters.
    """
    df = pd.read_csv(file_path)
    parameters = df.iloc[0].to_dict()  # Extract the first (and only) row as a dictionary
    return parameters



def run_forecast_BTU(EingabeZR, parameters, datapath="output", i18n=None):
    """
    Run BTU forecast analysis.
    
    This function replicates the R code for combining forecasts with a dynamic elastic net (DELNET)
    using a rolling window (e.g. 165 days). It performs data cleaning (rounding, negative clipping,
    missing-value interpolation, repeated‐value checking), fits an ElasticNetCV model on historical data,
    makes day‐ahead predictions, and then computes error measures (including monthly and yearly MSE, MAE, ME)
    and saves CSV files.
    
    Parameters:
      EingabeZR : pandas DataFrame containing the following columns:
          "A1", "A2", "A3", "A4", "A5", "A6", "A7"  – individual forecast providers
          "K"  – the 50Hertz combination forecast
          "HR" – actual (highly processed) values
          "Zeit" – time stamps
          "Jahr" – year
          "Monat" – month
          "MW_J" – installed wind capacity (first 9 rows used)
      parameters : dict with keys including:
          'd', 'e', 'FL_d', 'FX_h', 'AlphaParameter',
          'Anzahl.unmoeglich.max', 'Anzahl.leer.wdh.max', 'Anzahl.leer.gesamt.max',
          'Anzahl.Wertwdh.Null.max', 'Anzahl.Wertwdh.Wert.max',
          'Anzahl.unmoeglich.hist.max', 'Anzahl.leer.wdh.hist.max', 'Anzahl.leer.gesamt.hist.max',
          'Anzahl.Wertwdh.Null.hist.max', 'Anzahl.Wertwdh.Wert.hist.max',
          'Anzahl.Wertwdh.Wert.tol.hist.max'
      datapath : path to the folder where output CSV files will be written
      i18n : a dictionary for translations of file names (if None, default names are used)
      
    Returns:
      "DONE" if the procedure completes or an error message.
    """
    try:
        # Ensure the output folder exists.
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        # -----------------------------
        # 1. READ PARAMETERS AND DATA
        # ----------------------------- 
        e = int(parameters['e'])
        d = int(parameters['d'])
        FL_d = int(parameters['FL_d'])
        FX_h = int(parameters['FX_h'])
                 # forecast delivery time (e.g. 9:00) for day‐ahead forecasts
        AlphaParameter = parameters['AlphaParameter'] # elastic net mixing parameter

        # Cleaning parameters for the next–day forecasts:
        Anzahl_unmoeglich_max = parameters['Anzahl.unmoeglich.max']
        Anzahl_leer_wdh_max = parameters['Anzahl.leer.wdh.max']
        Anzahl_leer_gesamt_max = parameters['Anzahl.leer.gesamt.max']
        Anzahl_Wertwdh_Null_max = parameters['Anzahl.Wertwdh.Null.max']
        Anzahl_Wertwdh_Wert_max = parameters['Anzahl.Wertwdh.Wert.max']

        # Cleaning parameters for historical data:
        Anzahl_unmoeglich_hist_max = parameters['Anzahl.unmoeglich.hist.max']
        Anzahl_leer_wdh_hist_max = parameters['Anzahl.leer.wdh.hist.max']
        Anzahl_leer_gesamt_hist_max = parameters['Anzahl.leer.gesamt.hist.max']
        Anzahl_Wertwdh_Null_hist_max = parameters['Anzahl.Wertwdh.Null.hist.max']
        Anzahl_Wertwdh_Wert_hist_max = parameters['Anzahl.Wertwdh.Wert.hist.max']
        Anzahl_Wertwdh_Wert_tol_hist_max = parameters['Anzahl.Wertwdh.Wert.tol.hist.max']

        # Read and round the forecasts and actual values
        A1 = np.round(EingabeZR['A1'].astype(float), 0)
        A2 = np.round(EingabeZR['A2'].astype(float), 0)
        A3 = np.round(EingabeZR['A3'].astype(float), 0)
        A4 = np.round(EingabeZR['A4'].astype(float), 0)
        A5 = np.round(EingabeZR['A5'].astype(float), 0)
        A6 = np.round(EingabeZR['A6'].astype(float), 0)
        A7 = np.round(EingabeZR['A7'].astype(float), 0)
        K  = np.round(EingabeZR['K'].astype(float), 0)
        HR = np.round(EingabeZR['HR'].astype(float), 0)
        Z  = EingabeZR['Zeit'].values
        J  = EingabeZR['Jahr'].values
        M  = EingabeZR['Monat'].values
        MW_J = EingabeZR['MW_J'].iloc[:9].values  # installed capacity: first 9 values

        # Calculate the last quarter-hour index ("Ende")
        total_length = len(HR)
        Ende = total_length - 96 * (e - 1)

        # -----------------------------
        # 2. CLEAN THE ACTUAL VALUES (HR)
        # -----------------------------
        Max_install = np.nanmax(MW_J)
        HR_mean = np.nanmean(HR)
        HRN = np.zeros(Ende)
        # For each quarter-hour in HR (up to "Ende")
        for i in range(Ende):
            val = HR.iloc[i] if isinstance(HR, pd.Series) else HR[i]
            if not pd.isna(val):
                if 0 <= val <= Max_install:
                    HRN[i] = val
                else:
                    HRN[i] = HR_mean
            else:
                HRN[i] = 0

        # -----------------------------
        # 3. CALCULATE FORECAST ERRORS FOR INDIVIDUAL PROVIDERS
        # -----------------------------
        PF_K  = K[:Ende] - HRN
        PF_A1 = A1[:Ende] - HRN
        PF_A2 = A2[:Ende] - HRN
        PF_A3 = A3[:Ende] - HRN
        PF_A4 = A4[:Ende] - HRN
        PF_A5 = A5[:Ende] - HRN
        PF_A6 = A6[:Ende] - HRN
        PF_A7 = A7[:Ende] - HRN

        # -----------------------------
        # 4. SET UP ROLLING WINDOW
        # -----------------------------
        # 96 quarter-hours per day. The window length (in quarters) is:
        FL = FL_d * 96 - FX_h * 4
        # Fstart: first quarter-hour of the start day (Python uses 0-based indexing)
        Fstart = (d - 1) * 96  
        Fend = Fstart + FL       # window end index
        Pstart = FL_d * 96       # forecast start index for the next day

        # Allocate the combined forecast vector (BTU)
        BTU = np.full(Ende, np.nan)
        Zahl_Prognosen = 7  # maximum number of providers
        n_days = Ende // 96
        # Matrix for reasons why a provider is dropped (rows: days, cols: providers)
        Anbieter_stat_grund = np.zeros((n_days, Zahl_Prognosen))
        # Matrix to store model coefficients (first column = intercept, then 7 provider coefficients)
        Anbieter_Koeff = np.zeros((n_days, Zahl_Prognosen + 1))
        # To record the number of providers used each day
        Zahl_Anbieter_genutzt_tgl = np.zeros(n_days)

        # -----------------------------
        # 5. LOOP OVER DAYS TO CALCULATE THE DYNAMIC COMBINATION FORECAST
        # -----------------------------
        # Loop j over days for which new forecasts are available.
        # In R, j runs from (FL_d+d) to (Ende/96). Adjust for 0-based Python indexing.
        for j in range(FL_d + d - 1, n_days):
            Zahl_Anbieter_genutzt = 0

            # Extract the next-day forecasts (for day j) for each provider:
            start_idx = j * 96
            end_idx = (j + 1) * 96
            Prognosen_Folgetag = np.column_stack([
                A1.iloc[start_idx:end_idx].values,
                A2.iloc[start_idx:end_idx].values,
                A3.iloc[start_idx:end_idx].values,
                A4.iloc[start_idx:end_idx].values,
                A5.iloc[start_idx:end_idx].values,
                A6.iloc[start_idx:end_idx].values,
                A7.iloc[start_idx:end_idx].values
            ])
            # Historical forecasts used for model training:
            hist_start = (j - FL_d) * 96
            hist_end = hist_start + FL
            Prognosen_hist = np.column_stack([
                A1.iloc[hist_start:hist_end].values,
                A2.iloc[hist_start:hist_end].values,
                A3.iloc[hist_start:hist_end].values,
                A4.iloc[hist_start:hist_end].values,
                A5.iloc[hist_start:hist_end].values,
                A6.iloc[hist_start:hist_end].values,
                A7.iloc[hist_start:hist_end].values
            ])
            Istwerte_hist = HRN[hist_start:hist_end]

            # Process each provider (l = 0,...,6)
            for l in range(Zahl_Prognosen):
                B = Prognosen_Folgetag[:, l].copy()      # next-day forecast for provider l
                B_hist = Prognosen_hist[:, l].copy()       # historical forecast for provider l

                # Initialize counters for data cleaning
                Anzahl_unmoeglich = 0
                Anzahl_leer_wdh = 0
                Anzahl_leer_gesamt = 0
                Anzahl_Wertwdh_Null = 0
                Anzahl_Wertwdh_Wert = 0
                break_var = False

                # Loop over each quarter-hour of B (next-day forecast)
                for i_val in range(len(B)):
                    if not pd.isna(B[i_val]):
                        if B[i_val] < 0 or B[i_val] > Max_install:
                            Anzahl_unmoeglich += 1
                            if Anzahl_unmoeglich > Anzahl_unmoeglich_max:
                                B_hist = np.zeros_like(B_hist)
                                break_var = True
                                Anbieter_stat_grund[j, l] = 1
                                break
                    if i_val > 0 and (not pd.isna(B[i_val])) and (not pd.isna(B[i_val - 1])):
                        if B[i_val] == 0 and B[i_val] == B[i_val - 1]:
                            Anzahl_Wertwdh_Null += 1
                            if Anzahl_Wertwdh_Null > Anzahl_Wertwdh_Null_max:
                                B_hist = np.zeros_like(B_hist)
                                break_var = True
                                Anbieter_stat_grund[j, l] = 2
                                break
                        else:
                            Anzahl_Wertwdh_Null = 0
                        if B[i_val] != 0 and B[i_val] == B[i_val - 1]:
                            Anzahl_Wertwdh_Wert += 1
                            if Anzahl_Wertwdh_Wert > Anzahl_Wertwdh_Wert_max:
                                B_hist = np.zeros_like(B_hist)
                                break_var = True
                                Anbieter_stat_grund[j, l] = 3
                                break
                        else:
                            Anzahl_Wertwdh_Wert = 0
                    else:
                        Anzahl_Wertwdh_Null = 0
                        Anzahl_Wertwdh_Wert = 0
                    if pd.isna(B[i_val]):
                        Anzahl_leer_wdh += 1
                        Anzahl_leer_gesamt += 1
                        if (Anzahl_leer_wdh > Anzahl_leer_wdh_max or 
                            Anzahl_leer_gesamt > Anzahl_leer_gesamt_max):
                            B_hist = np.zeros_like(B_hist)
                            break_var = True
                            Anbieter_stat_grund[j, l] = 4
                            break
                    else:
                        Anzahl_leer_wdh = 0

                # If the next-day forecast passed the plausibility test, clean its unplausible values
                if not break_var:
                    for i_val in range(len(B)):
                        if not pd.isna(B[i_val]) and (B[i_val] < 0 or B[i_val] > Max_install):
                            B[i_val] = np.nan
                    # Interpolate if at least 2 valid points exist; otherwise use zeros
                    if np.sum(~np.isnan(B)) >= 2:
                        B = pd.Series(B).interpolate(method='linear', limit_direction='both').values
                    else:
                        B = np.zeros_like(B)

                # Now, process the historical forecast (B_hist) with similar checks:
                Anzahl_unmoeglich_hist = 0
                Anzahl_leer_wdh_hist = 0
                Anzahl_leer_gesamt_hist = 0
                Anzahl_Wertwdh_Null_hist = 0
                Anzahl_Wertwdh_Wert_hist = 0
                break_var_hist = False

                if not break_var:
                    for i_val in range(len(B_hist)):
                        if not pd.isna(B_hist[i_val]):
                            if B_hist[i_val] < 0 or B_hist[i_val] > Max_install:
                                Anzahl_unmoeglich_hist += 1
                                if Anzahl_unmoeglich_hist > Anzahl_unmoeglich_hist_max:
                                    B_hist = np.zeros_like(B_hist)
                                    break_var_hist = True
                                    Anbieter_stat_grund[j, l] = 5
                                    break
                        if i_val > 0 and (not pd.isna(B_hist[i_val])) and (not pd.isna(B_hist[i_val - 1])):
                            if B_hist[i_val] == 0 and B_hist[i_val] == B_hist[i_val - 1]:
                                Anzahl_Wertwdh_Null_hist += 1
                                if Anzahl_Wertwdh_Null_hist > Anzahl_Wertwdh_Null_hist_max:
                                    B_hist = np.zeros_like(B_hist)
                                    break_var_hist = True
                                    Anbieter_stat_grund[j, l] = 6
                                    break
                            else:
                                Anzahl_Wertwdh_Null_hist = 0
                            if B_hist[i_val] != 0 and B_hist[i_val] == B_hist[i_val - 1]:
                                Anzahl_Wertwdh_Wert_hist += 1
                                if Anzahl_Wertwdh_Wert_hist > Anzahl_Wertwdh_Wert_hist_max:
                                    B_hist = np.zeros_like(B_hist)
                                    break_var_hist = True
                                    Anbieter_stat_grund[j, l] = 7
                                    break
                            else:
                                # Tolerate a short run of equal values
                                if Anzahl_Wertwdh_Wert_hist > Anzahl_Wertwdh_Wert_tol_hist_max:
                                    B_hist[i_val - Anzahl_Wertwdh_Wert_hist:i_val] = 0
                                Anzahl_Wertwdh_Wert_hist = 0
                        else:
                            if i_val > 0 and Anzahl_Wertwdh_Wert_hist > Anzahl_Wertwdh_Wert_tol_hist_max:
                                B_hist[i_val - Anzahl_Wertwdh_Wert_hist:i_val] = 0
                                Anzahl_Wertwdh_Wert_hist = 0
                            else:
                                Anzahl_Wertwdh_Null_hist = 0
                                Anzahl_Wertwdh_Wert_hist = 0
                        if pd.isna(B_hist[i_val]):
                            Anzahl_leer_wdh_hist += 1
                            Anzahl_leer_gesamt_hist += 1
                            if (Anzahl_leer_wdh_hist > Anzahl_leer_wdh_hist_max or 
                                Anzahl_leer_gesamt_hist > Anzahl_leer_gesamt_hist_max):
                                B_hist = np.zeros_like(B_hist)
                                break_var_hist = True
                                Anbieter_stat_grund[j, l] = 8
                                break
                        else:
                            Anzahl_leer_wdh_hist = 0

                if not break_var_hist:
                    for i_val in range(len(B_hist)):
                        if not pd.isna(B_hist[i_val]) and (B_hist[i_val] < 0 or B_hist[i_val] > Max_install):
                            B_hist[i_val] = np.nan
                    if np.sum(~np.isnan(B_hist)) >= 2:
                        B_hist = pd.Series(B_hist).interpolate(method='linear', limit_direction='both').values
                    else:
                        B_hist = np.zeros_like(B_hist)

                # If both next–day and historical data passed the cleaning, count this provider as used.
                if not break_var and not break_var_hist:
                    Zahl_Anbieter_genutzt += 1

                # Update the matrices with the cleaned provider forecasts
                Prognosen_Folgetag[:, l] = B
                Prognosen_hist[:, l] = B_hist

            # Record the number of providers used on day j
            Zahl_Anbieter_genutzt_tgl[j] = Zahl_Anbieter_genutzt

            # -----------------------------
            # 6. FIT THE ELASTIC NET MODEL ON THE HISTORICAL DATA
            # -----------------------------

            alphas = np.logspace(-4, 4, 100)

            model = ElasticNetCV(
                l1_ratio=AlphaParameter,     # corresponds to alpha in glmnet (0=ridge, 1=lasso)
                fit_intercept=True,          # glmnet always includes intercept by default
                cv=10,                       # number of folds
                random_state=0,
                selection='cyclic',          # default, same as glmnet unless you're doing strong rules
                n_alphas=100,                # resolution of the lambda path (can increase for closer match)
                #alphas=alphas,
                tol=1e-4                     # convergence tolerance (default, can tweak if needed)
            )

            # Initialize scaler and fit on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(Prognosen_hist)

            # Fit model on scaled data
            model.fit(X_train_scaled, Istwerte_hist)

            #model.fit(Prognosen_hist, Istwerte_hist)
            # In scikit-learn the chosen regularization parameter is model.alpha_
            LambdaParameter = model.alpha_
            # Save the coefficients (intercept and provider coefficients)
            coef_all = np.hstack([model.intercept_, model.coef_])
            Anbieter_Koeff[j, :] = coef_all

            # -----------------------------
            # 7. PREDICT THE BTU COMBINED FORECAST FOR THE DAY
            # -----------------------------
            for r in range(96):
                x_new = Prognosen_Folgetag[r, :].reshape(1, -1)
                x_new_scaled = scaler.transform(x_new)
                pred = model.predict(x_new_scaled)[0]
                # Round the prediction and assign it to the proper index in BTU
                BTU[j * 96 + r] = np.round(pred, 0)

        # -----------------------------
        # 8. POST–PROCESSING
        # -----------------------------
        # Correct any negative BTU values (clip at 0) and compute the forecast error PF_BTU.
        PF_BTU = np.empty(Ende)
        for i in range(Pstart, Ende):
            if not pd.isna(BTU[i]) and BTU[i] < 0:
                BTU[i] = 0
            PF_BTU[i] = BTU[i] - HRN[i]

        # -----------------------------
        # 9. CALCULATE AVERAGE (AM) FORECAST AND ITS ERROR
        # -----------------------------
        Anbieteranzahl = np.zeros(len(HR))
        for i in range(len(HR)):
            count = 0
            s = 0
            for arr in [A1, A2, A3, A4, A5, A6, A7]:
                if not pd.isna(arr.iloc[i]):
                    count += 1
                    s += arr.iloc[i]
            Anbieteranzahl[i] = count

        AM = np.full(Ende, np.nan)
        A_total = np.full(Ende, np.nan)
        for i in range(Pstart, Ende):
            s = 0
            for arr in [A1, A2, A3, A4, A5, A6, A7]:
                s += 0 if pd.isna(arr.iloc[i]) else arr.iloc[i]
            A_total[i] = s
            if Anbieteranzahl[i] > 0:
                AM[i] = np.round(s / Anbieteranzahl[i])
        PF_AM = AM - HRN

        # -----------------------------
        # 10. OVERALL ERROR MEASURES
        # -----------------------------
        mse_AM = np.round(np.nanmean(PF_AM[:Ende] ** 2), 0)
        mae_AM = np.round(np.nanmean(np.abs(PF_AM[:Ende])), 0)
        me_AM  = np.round(np.nanmean(PF_AM[:Ende]), 0)

        mse_K = np.round(np.nanmean(PF_K[:Ende] ** 2), 0)
        mae_K = np.round(np.nanmean(np.abs(PF_K[:Ende])), 0)
        me_K  = np.round(np.nanmean(PF_K[:Ende]), 0)

        mse_BTU = np.round(np.nanmean(PF_BTU[:Ende] ** 2), 0)
        mae_BTU = np.round(np.nanmean(np.abs(PF_BTU[:Ende])), 0)
        me_BTU  = np.round(np.nanmean(PF_BTU[:Ende]), 0)

        mse_A1 = np.round(np.nanmean(PF_A1[:Ende] ** 2), 0)
        mae_A1 = np.round(np.nanmean(np.abs(PF_A1[:Ende])), 0)
        me_A1  = np.round(np.nanmean(PF_A1[:Ende]), 0)

        mse_A2 = np.round(np.nanmean(PF_A2[:Ende] ** 2), 0)
        mae_A2 = np.round(np.nanmean(np.abs(PF_A2[:Ende])), 0)
        me_A2  = np.round(np.nanmean(PF_A2[:Ende]), 0)

        mse_A3 = np.round(np.nanmean(PF_A3[:Ende] ** 2), 0)
        mae_A3 = np.round(np.nanmean(np.abs(PF_A3[:Ende])), 0)
        me_A3  = np.round(np.nanmean(PF_A3[:Ende]), 0)

        mse_A4 = np.round(np.nanmean(PF_A4[:Ende] ** 2), 0)
        mae_A4 = np.round(np.nanmean(np.abs(PF_A4[:Ende])), 0)
        me_A4  = np.round(np.nanmean(PF_A4[:Ende]), 0)

        mse_A5 = np.round(np.nanmean(PF_A5[:Ende] ** 2), 0)
        mae_A5 = np.round(np.nanmean(np.abs(PF_A5[:Ende])), 0)
        me_A5  = np.round(np.nanmean(PF_A5[:Ende]), 0)

        mse_A6 = np.round(np.nanmean(PF_A6[:Ende] ** 2), 0)
        mae_A6 = np.round(np.nanmean(np.abs(PF_A6[:Ende])), 0)
        me_A6  = np.round(np.nanmean(PF_A6[:Ende]), 0)

        mse_A7 = np.round(np.nanmean(PF_A7[:Ende] ** 2), 0)
        mae_A7 = np.round(np.nanmean(np.abs(PF_A7[:Ende])), 0)
        me_A7  = np.round(np.nanmean(PF_A7[:Ende]), 0)

        # -----------------------------
        # 11. SAVE FORECAST ERROR DATA TO CSV FILES
        # -----------------------------
        # Use i18n dictionary if provided for file names.
        Prognosefehler_anb = pd.DataFrame({
            "Z": Z[:Ende],
            "PF_A1": PF_A1[:Ende],
            "PF_A2": PF_A2[:Ende],
            "PF_A3": PF_A3[:Ende],
            "PF_A4": PF_A4[:Ende],
            "PF_A5": PF_A5[:Ende],
            "PF_A6": PF_A6[:Ende],
            "PF_A7": PF_A7[:Ende]
        })
        Prognosefehler_anb_filename = f"{i18n['Forecast_Error_Provider']}.csv" if i18n else "Forecast_Error_Provider.csv"
        Prognosefehler_anb.to_csv(os.path.join(datapath, Prognosefehler_anb_filename), index=False, na_rep="#NV")

        Prognosefehler_ges = pd.DataFrame({
            "Z": Z[:Ende],
            "Anbieteranzahl": Anbieteranzahl[:Ende],
            "AM": AM[:Ende],
            "K": K[:Ende],
            "BTU": BTU[:Ende],
            "PF_AM": PF_AM[:Ende],
            "PF_K": PF_K[:Ende],
            "PF_BTU": PF_BTU[:Ende]
        })
        Prognosefehler_ges_filename = f"{i18n['Forecast_Error_Total']}.csv" if i18n else "Forecast_Error_Total.csv"
        Prognosefehler_ges.to_csv(os.path.join(datapath, Prognosefehler_ges_filename), index=False, na_rep="#NV")

        Fehlermasse = pd.DataFrame({
            "Error_Type": ["MSE", "MAE", "ME"],
            "AM": [mse_AM, mae_AM, me_AM],
            "K": [mse_K, mae_K, me_K],
            "BTU": [mse_BTU, mae_BTU, me_BTU],
            "A1": [mse_A1, mae_A1, me_A1],
            "A2": [mse_A2, mae_A2, me_A2],
            "A3": [mse_A3, mae_A3, me_A3],
            "A4": [mse_A4, mae_A4, me_A4],
            "A5": [mse_A5, mae_A5, me_A5],
            "A6": [mse_A6, mae_A6, me_A6],
            "A7": [mse_A7, mae_A7, me_A7]
        })
        Fehlermasse_filename = f"{i18n['Error_Mass']}.csv" if i18n else "Error_Mass.csv"
        Fehlermasse.to_csv(os.path.join(datapath, Fehlermasse_filename), index=False, na_rep="#NV")
        
        # -----------------------------
        # 12. CALCULATE YEARLY ERROR MEASURES
        # -----------------------------
        # Determine yearly boundaries.
        # (Here we assume J (year) is valid for the first Ende entries.)
        J_subset = J[:Ende]
        JAnz = int(np.max(J_subset) - np.min(J_subset) + 1)
        JBeg = np.ones(JAnz + 1, dtype=int)
        JBeg[-1] = Ende + 1  # last index (R uses 1-based indexing)
        JLis = np.full(JAnz, J_subset[0])
        JNum = 1  # starting from index 1 (Python: 0-based, but we mimic R's 1-based logic here)
        for i in range(1, Ende):
            if J_subset[i] != J_subset[i - 1]:
                if JNum < JAnz:
                    JBeg[JNum] = i + 1  # convert to 1-based indexing
                    JLis[JNum] = J_subset[i]
                    JNum += 1
        
        # Initialize arrays for yearly error metrics
        mse_AM_J = np.empty(JAnz)
        mse_K_J = np.empty(JAnz)
        mse_BTU_J = np.empty(JAnz)
        mae_AM_J = np.empty(JAnz)
        mae_K_J = np.empty(JAnz)
        mae_BTU_J = np.empty(JAnz)
        me_AM_J = np.empty(JAnz)
        me_K_J = np.empty(JAnz)
        me_BTU_J = np.empty(JAnz)
        mse_A1_J = np.empty(JAnz)
        mse_A2_J = np.empty(JAnz)
        mse_A3_J = np.empty(JAnz)
        mse_A4_J = np.empty(JAnz)
        mse_A5_J = np.empty(JAnz)
        mse_A6_J = np.empty(JAnz)
        mse_A7_J = np.empty(JAnz)
        mae_A1_J = np.empty(JAnz)
        mae_A2_J = np.empty(JAnz)
        mae_A3_J = np.empty(JAnz)
        mae_A4_J = np.empty(JAnz)
        mae_A5_J = np.empty(JAnz)
        mae_A6_J = np.empty(JAnz)
        mae_A7_J = np.empty(JAnz)
        me_A1_J = np.empty(JAnz)
        me_A2_J = np.empty(JAnz)
        me_A3_J = np.empty(JAnz)
        me_A4_J = np.empty(JAnz)
        me_A5_J = np.empty(JAnz)
        me_A6_J = np.empty(JAnz)
        me_A7_J = np.empty(JAnz)
        
        # Loop over each year (using the indices in JBeg; note the conversion from 1-based to 0-based)
        for l in range(JAnz):
            start = JBeg[l] - 1
            end = JBeg[l + 1] - 1  # end index is exclusive
            mse_AM_J[l]  = np.round(np.nanmean(PF_AM[start:end] ** 2), 0)
            mae_AM_J[l]  = np.round(np.nanmean(np.abs(PF_AM[start:end])), 0)
            me_AM_J[l]   = np.round(np.nanmean(PF_AM[start:end]), 0)
            mse_K_J[l]   = np.round(np.nanmean(PF_K[start:end] ** 2), 0)
            mae_K_J[l]   = np.round(np.nanmean(np.abs(PF_K[start:end])), 0)
            me_K_J[l]    = np.round(np.nanmean(PF_K[start:end]), 0)
            mse_BTU_J[l] = np.round(np.nanmean(PF_BTU[start:end] ** 2), 0)
            mae_BTU_J[l] = np.round(np.nanmean(np.abs(PF_BTU[start:end])), 0)
            me_BTU_J[l]  = np.round(np.nanmean(PF_BTU[start:end]), 0)
            mse_A1_J[l]  = np.round(np.nanmean(PF_A1[start:end] ** 2), 0)
            mae_A1_J[l]  = np.round(np.nanmean(np.abs(PF_A1[start:end])), 0)
            me_A1_J[l]   = np.round(np.nanmean(PF_A1[start:end]), 0)
            mse_A2_J[l]  = np.round(np.nanmean(PF_A2[start:end] ** 2), 0)
            mae_A2_J[l]  = np.round(np.nanmean(np.abs(PF_A2[start:end])), 0)
            me_A2_J[l]   = np.round(np.nanmean(PF_A2[start:end]), 0)
            mse_A3_J[l]  = np.round(np.nanmean(PF_A3[start:end] ** 2), 0)
            mae_A3_J[l]  = np.round(np.nanmean(np.abs(PF_A3[start:end])), 0)
            me_A3_J[l]   = np.round(np.nanmean(PF_A3[start:end]), 0)
            mse_A4_J[l]  = np.round(np.nanmean(PF_A4[start:end] ** 2), 0)
            mae_A4_J[l]  = np.round(np.nanmean(np.abs(PF_A4[start:end])), 0)
            me_A4_J[l]   = np.round(np.nanmean(PF_A4[start:end]), 0)
            mse_A5_J[l]  = np.round(np.nanmean(PF_A5[start:end] ** 2), 0)
            mae_A5_J[l]  = np.round(np.nanmean(np.abs(PF_A5[start:end])), 0)
            me_A5_J[l]   = np.round(np.nanmean(PF_A5[start:end]), 0)
            mse_A6_J[l]  = np.round(np.nanmean(PF_A6[start:end] ** 2), 0)
            mae_A6_J[l]  = np.round(np.nanmean(np.abs(PF_A6[start:end])), 0)
            me_A6_J[l]   = np.round(np.nanmean(PF_A6[start:end]), 0)
            mse_A7_J[l]  = np.round(np.nanmean(PF_A7[start:end] ** 2), 0)
            mae_A7_J[l]  = np.round(np.nanmean(np.abs(PF_A7[start:end])), 0)
            me_A7_J[l]   = np.round(np.nanmean(PF_A7[start:end]), 0)
        
        # Combine yearly error measures into a DataFrame.
        # Note: The R code includes additional columns like "installierte Leistung [MW]",
        # "mittlere Einspeisung [MW]", and "Anbieteranzahl" that depend on other computations.
        # Here we add placeholders (or simple approximations) for these.
        Fehlermasse_J = pd.DataFrame({
            "Jahr": JLis,
            "installierte Leistung [MW]": MW_J[:JAnz] if len(MW_J) >= JAnz else np.nan,
            "mittlere Einspeisung [MW]": np.nan,  # (JMW not computed in this snippet)
            "Anbieteranzahl": np.nan,              # (yearly average number of providers; compute as needed)
            "mse_AM_J": mse_AM_J,
            "mse_K_J": mse_K_J,
            "mse_BTU_J": mse_BTU_J,
            "mae_AM_J": mae_AM_J,
            "mae_K_J": mae_K_J,
            "mae_BTU_J": mae_BTU_J,
            "me_AM_J": me_AM_J,
            "me_K_J": me_K_J,
            "me_BTU_J": me_BTU_J,
            "me_A1_J": me_A1_J,
            "me_A2_J": me_A2_J,
            "me_A3_J": me_A3_J,
            "me_A4_J": me_A4_J,
            "me_A5_J": me_A5_J,
            "me_A6_J": me_A6_J,
            "me_A7_J": me_A7_J,
            "mae_A1_J": mae_A1_J,
            "mae_A2_J": mae_A2_J,
            "mae_A3_J": mae_A3_J,
            "mae_A4_J": mae_A4_J,
            "mae_A5_J": mae_A5_J,
            "mae_A6_J": mae_A6_J,
            "mae_A7_J": mae_A7_J,
            "mse_A1_J": mse_A1_J,
            "mse_A2_J": mse_A2_J,
            "mse_A3_J": mse_A3_J,
            "mse_A4_J": mse_A4_J,
            "mse_A5_J": mse_A5_J,
            "mse_A6_J": mse_A6_J,
            "mse_A7_J": mse_A7_J
        })
        Fehlermasse_J_filename = f"{i18n['Error_Measures_Yearly']}.csv" if i18n else "Error_Measures_Yearly.csv"
        Fehlermasse_J.to_csv(os.path.join(datapath, Fehlermasse_J_filename), index=False, na_rep="#NV")
        
        # -----------------------------
        # 13. MONTHLY ERROR MEASURES AND CHART DATA
        # -----------------------------
        # (A similar procedure applies to calculate monthly error measures.
        # Here we provide placeholders; you would need to compute MBeg and MLis from M (month)
        # and then compute metrics analogous to the yearly ones.)
        # For example:
        # MBeg, MLis, and monthly error arrays (mse_AM_M, mae_AM_M, etc.) would be computed here.
        # Then save them to CSV:
        #
        # Fehlermasse_M = pd.DataFrame({ ... })
        # Fehlermasse_M_filename = f"{i18n['Error_Measures_Monthly']}.csv" if i18n else "Error_Measures_Monthly.csv"
        # Fehlermasse_M.to_csv(os.path.join(datapath, Fehlermasse_M_filename), index=False, na_rep="#NV")
        #
        # Similarly, chart data for Grafik1, Grafik2, ..., Grafik6 are computed.
        
        # Example for Grafik1 (Yearly Chart Data):
        Grafik1 = pd.DataFrame({
            "Jahr": JLis,
            "mse_A1_J": mse_A1_J,
            "mse_A2_J": mse_A2_J,
            "mse_A3_J": mse_A3_J,
            "mse_A4_J": mse_A4_J,
            "mse_A5_J": mse_A5_J,
            "mse_A6_J": mse_A6_J,
            "mse_A7_J": mse_A7_J,
            "MW_J": MW_J[:JAnz] if len(MW_J) >= JAnz else np.nan,
            "JMW": np.nan  # (JMW not computed in this snippet)
        })
        grafik1_filename = f"{i18n['Chart1']}.csv" if i18n else "Chart1.csv"
        Grafik1.to_csv(os.path.join(datapath, grafik1_filename), index=False, na_rep="#NV")
        
        # Example for Grafik2 (Provider MSE divided by average generation):
        Grafik2 = pd.DataFrame({
            "Jahr": JLis,
            "mse_A1_J/JMW": np.round(mse_A1_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A2_J/JMW": np.round(mse_A2_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A3_J/JMW": np.round(mse_A3_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A4_J/JMW": np.round(mse_A4_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A5_J/JMW": np.round(mse_A5_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A6_J/JMW": np.round(mse_A6_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0),
            "mse_A7_J/JMW": np.round(mse_A7_J / np.where(np.array(MW_J[:JAnz]) == 0, np.nan, MW_J[:JAnz]), 0)
        })
        grafik2_filename = f"{i18n['Chart2']}.csv" if i18n else "Chart2.csv"
        Grafik2.to_csv(os.path.join(datapath, grafik2_filename), index=False, na_rep="#NV")
        
        # (Similar placeholders can be set up for Grafik3, Grafik4, Grafik5, Grafik6.)

        # -----------------------------
        # 13. MONTHLY ERROR MEASURES
        # -----------------------------
        M_subset = M[:Ende]
        MJLis = []
        MBeg = []
        prev_year, prev_month = J[0], M[0]

        # Create list of tuples for year+month combinations
        for i in range(1, Ende):
            if M[i] != prev_month or J[i] != prev_year:
                MJLis.append((prev_year, prev_month))
                MBeg.append(i)
                prev_year, prev_month = J[i], M[i]
        # Add last month
        MJLis.append((prev_year, prev_month))
        MBeg.append(Ende)

        mse_AM_M, mse_K_M, mse_BTU_M = [], [], []
        Anbieteranzahl_M = []

        for idx in range(len(MJLis)):
            start = MBeg[idx - 1] if idx > 0 else 0
            end = MBeg[idx]

            mse_AM_M.append(np.round(np.nanmean(PF_AM[start:end] ** 2), 0))
            mse_K_M.append(np.round(np.nanmean(PF_K[start:end] ** 2), 0))
            mse_BTU_M.append(np.round(np.nanmean(PF_BTU[start:end] ** 2), 0))
            Anbieteranzahl_M.append(int(np.nanmean(Anbieteranzahl[start:end])))

        Fehlermasse_M = pd.DataFrame({
            "Jahr": [jm[0] for jm in MJLis],
            "Monat": [jm[1] for jm in MJLis],
            "mse_AM": mse_AM_M,
            "mse_K": mse_K_M,
            "mse_BTU": mse_BTU_M,
            "Anbieteranzahl": Anbieteranzahl_M
        })

        Fehlermasse_M_filename = f"{i18n['Error_Measures_Monthly']}.csv" if i18n else "Error_Measures_Monthly.csv"
        Fehlermasse_M.to_csv(os.path.join(datapath, Fehlermasse_M_filename), index=False, na_rep="#NV")

        # -----------------------------
        # 14. SAVE PROVIDER COEFFICIENTS
        # -----------------------------
        Anbieter_Koeff_filename = f"{i18n['AK']}.csv" if i18n else "AK.csv"
        # Save only the coefficients from days starting at (FL_d+d)
        pd.DataFrame(Anbieter_Koeff[FL_d + d - 1:]).to_csv(os.path.join(datapath, Anbieter_Koeff_filename), index=False, na_rep="#NV")
        
        # -----------------------------
        # 15. DESCRIPTIVE STATISTICS BY YEAR
        # -----------------------------
        sd_HR_J = np.empty(JAnz)
        me_HR_J_arr = np.empty(JAnz)
        sd_K_J = np.empty(JAnz)
        me_K_J_arr = np.empty(JAnz)
        sd_A1_J = np.empty(JAnz)
        me_A1_J_arr = np.empty(JAnz)
        sd_A2_J = np.empty(JAnz)
        me_A2_J_arr = np.empty(JAnz)
        sd_A3_J = np.empty(JAnz)
        me_A3_J_arr = np.empty(JAnz)
        sd_A4_J = np.empty(JAnz)
        me_A4_J_arr = np.empty(JAnz)
        sd_A5_J = np.empty(JAnz)
        me_A5_J_arr = np.empty(JAnz)
        sd_A6_J = np.empty(JAnz)
        me_A6_J_arr = np.empty(JAnz)
        sd_A7_J = np.empty(JAnz)
        me_A7_J_arr = np.empty(JAnz)
        
        for l in range(JAnz):
            start = JBeg[l] - 1
            end = JBeg[l + 1] - 1 if l + 1 < len(JBeg) else Ende
            sd_HR_J[l] = np.round(np.nanstd(HR[:Ende][start:end]), 0)
            me_HR_J_arr[l] = np.round(np.nanmean(HR[:Ende][start:end]), 0)
            sd_K_J[l] = np.round(np.nanstd(K[:Ende][start:end]), 0)
            me_K_J_arr[l] = np.round(np.nanmean(K[:Ende][start:end]), 0)
            sd_A1_J[l] = np.round(np.nanstd(A1.iloc[start:end]), 0)
            me_A1_J_arr[l] = np.round(np.nanmean(A1.iloc[start:end]), 0)
            sd_A2_J[l] = np.round(np.nanstd(A2.iloc[start:end]), 0)
            me_A2_J_arr[l] = np.round(np.nanmean(A2.iloc[start:end]), 0)
            sd_A3_J[l] = np.round(np.nanstd(A3.iloc[start:end]), 0)
            me_A3_J_arr[l] = np.round(np.nanmean(A3.iloc[start:end]), 0)
            sd_A4_J[l] = np.round(np.nanstd(A4.iloc[start:end]), 0)
            me_A4_J_arr[l] = np.round(np.nanmean(A4.iloc[start:end]), 0)
            sd_A5_J[l] = np.round(np.nanstd(A5.iloc[start:end]), 0)
            me_A5_J_arr[l] = np.round(np.nanmean(A5.iloc[start:end]), 0)
            sd_A6_J[l] = np.round(np.nanstd(A6.iloc[start:end]), 0)
            me_A6_J_arr[l] = np.round(np.nanmean(A6.iloc[start:end]), 0)
            sd_A7_J[l] = np.round(np.nanstd(A7.iloc[start:end]), 0)
            me_A7_J_arr[l] = np.round(np.nanmean(A7.iloc[start:end]), 0)
        
        Descript_Stat = pd.DataFrame({
            "Jahr": JLis,
            "me_HR_J": me_HR_J_arr,
            "me_K_J": me_K_J_arr,
            "me_A1_J": me_A1_J_arr,
            "me_A2_J": me_A2_J_arr,
            "me_A3_J": me_A3_J_arr,
            "me_A4_J": me_A4_J_arr,
            "me_A5_J": me_A5_J_arr,
            "me_A6_J": me_A6_J_arr,
            "me_A7_J": me_A7_J_arr,
            "sd_HR_J": sd_HR_J,
            "sd_K_J": sd_K_J,
            "sd_A1_J": sd_A1_J,
            "sd_A2_J": sd_A2_J,
            "sd_A3_J": sd_A3_J,
            "sd_A4_J": sd_A4_J,
            "sd_A5_J": sd_A5_J,
            "sd_A6_J": sd_A6_J,
            "sd_A7_J": sd_A7_J
        })
        Descriptive_Statistics_filename = f"{i18n['Descriptive_Statistics']}.csv" if i18n else "Descriptive_Statistics.csv"
        Descript_Stat.to_csv(os.path.join(datapath, Descriptive_Statistics_filename), index=False, na_rep="#NV")
        
        # -----------------------------
        # 16. COMPUTE QUANTILE LIMITS FOR HR (IF NEEDED)
        # -----------------------------
        S_sorted = np.sort(HRN)
        quantil_lim_1_3 = S_sorted[int(np.round(len(S_sorted) / 3))]
        quantil_lim_2_3 = S_sorted[int(np.round(2 * len(S_sorted) / 3))]
        
        return "DONE"
    except Exception as e:
        return str(e)




if __name__ == "__main__":
    # Example usage
    print("Running forecast, starting with data loading...")
    data = load_forecast_data("data/data_files/input_files/sample_complete_Ty_fixed_rounded.csv")
    print('Data loaded, loading forecasts...')
    parameters = load_parameters("data/data_files/input_files/params.csv")
    print('Parameters loaded, running forecast...')
    result = run_forecast_BTU(data, parameters)
    print(result)



