import streamlit as st
import streamlit_authenticator as stauth
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from openpyxl import Workbook
import pickle

# --- USER AUTHENTICATION ---
names = ["Avidity Associate"]
usernames = ["Avidity"]

#----Load hashed passwords----
file_path = Path(r'C:\Users\Admin\Desktop\Potency_app\generate_keys.py').parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names=names,
    usernames=usernames,
    passwords=hashed_passwords,
    cookie_name="Potency Suit",
    key="abcdef",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login", "sidebar")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:

    # -- Script for system and sample suit --

    # Function to calculate %CV
    def calculate_cv(data):
        mean = np.nanmean(data, axis=1)
        std_dev = np.nanstd(data, axis=1, ddof=1)
        cv = (std_dev / mean) * 100
        return cv

    # Define the 4PL function
    def four_param_logistic(x, a, b, c, d):
        return ((a - d) / (1.0 + np.power(x / c, b))) + d

    # Function to fit model and calculate R-squared and parameters
    def fit_and_calculate_r2(x, y):
        popt, _ = curve_fit(four_param_logistic, x, y, maxfev=10000)
        y_pred = four_param_logistic(x, *popt)
        r2 = r2_score(y, y_pred)
        return round(r2, 2), [round(param, 5) for param in popt]

    # Function to plot 4PL curves with error bars on a single graph
    def plot_combined_4pl_curves(nM, rs_log_mean, rs_params, rs_log_std, 
                                iac_log_mean, iac_params, iac_log_std, 
                                ts_log_mean, ts_params, ts_log_std):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot RS Data
        ax.errorbar(nM, rs_log_mean, yerr=rs_log_std, fmt='o', label='RS Data', color='blue', ecolor='lightblue', elinewidth=2, capsize=3)
        x_fit = np.logspace(np.log10(min(nM)), np.log10(max(nM)), 100)
        y_fit = four_param_logistic(x_fit, *rs_params)
        ax.plot(x_fit, y_fit, label='RS 4PL Fit', color='blue')
        
        # Plot IAC Data
        ax.errorbar(nM, iac_log_mean, yerr=iac_log_std, fmt='o', label='IAC Data', color='green', ecolor='lightgreen', elinewidth=2, capsize=3)
        y_fit = four_param_logistic(x_fit, *iac_params)
        ax.plot(x_fit, y_fit, label='IAC 4PL Fit', color='green')
        
        # Plot TS Data
        ax.errorbar(nM, ts_log_mean, yerr=ts_log_std, fmt='o', label='TS Data', color='red', ecolor='lightcoral', elinewidth=2, capsize=3)
        y_fit = four_param_logistic(x_fit, *ts_params)
        ax.plot(x_fit, y_fit, label='TS 4PL Fit', color='red')
        
        # Set scale, labels, title, and legend
        ax.set_xscale('log')
        ax.set_xlabel('Dose (nM)')
        ax.set_ylabel('Response (log10)')
        ax.legend()
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
     # Dixon's Q test for outlier detection and removal
    def dixon_outlier_removal(data, q_crit_3=0.970, q_crit_4=0.829):
        def get_q_value(vals):
            sorted_vals = sorted(vals)
            Q = np.abs(sorted_vals[-1] - sorted_vals[-2]) / (max(sorted_vals) - min(sorted_vals))
            return Q

        if len(data) not in [3, 4]:
            return data  # Only applicable for 3 or 4 replicates

        q_value = get_q_value(data)
        q_critical = q_crit_3 if len(data) == 3 else q_crit_4

        if q_value > q_critical:
            sorted_data = sorted(data)
            outlier = sorted_data[-1]
            data = [x for x in data if x != outlier]
            data.append(np.nan)  # Pad with NaN to maintain array shape
        
        return data

    # Load the Excel file
    st.title('Potency System Suit and Sample Acceptance :sunglasses:')

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", sheet_names)

        df = pd.read_excel(uploaded_file, sheet_name=page)

        # Extract the nM concentrations
        nM = df['nM'].values

        # Determine the number of replicates
        replicate_columns = [col for col in df.columns if col.startswith('RS_') or col.startswith('IAC_') or col.startswith('TS_')]
        num_replicates = len(replicate_columns) // 3  # Assuming equal number of RS, IAC, and TS columns

        # Extract the data for RS, IAC, and TS based on the number of replicates
        rs_data = df[[f'RS_{i+1}' for i in range(num_replicates)]].values
        iac_data = df[[f'IAC_{i+1}' for i in range(num_replicates)]].values
        ts_data = df[[f'TS_{i+1}' for i in range(num_replicates)]].values

        # Apply Dixon's outlier removal
        #rs_data = np.apply_along_axis(dixon_outlier_removal, 1, rs_data)
        #iac_data = np.apply_along_axis(dixon_outlier_removal, 1, iac_data)
        #ts_data = np.apply_along_axis(dixon_outlier_removal, 1, ts_data)

        # Calculate %CV for RF, IAC, and TS
        rs_cv = calculate_cv(rs_data)
        iac_cv = calculate_cv(iac_data)
        ts_cv = calculate_cv(ts_data)

        cv_df =pd.DataFrame({
            'CV_RS': np.round(rs_cv, 2),
            'CV_IAC': np.round(iac_cv, 2),
            'CV_TS': np.round(ts_cv, 2)
        })
    
        st.write("Coefficient of Variation (CV) Data")
        st.dataframe(cv_df)

        rs_log = np.log10(rs_data)
        iac_log = np.log10(iac_data)
        ts_log = np.log10(ts_data)

        # Prepare data and calculate R-squared and parameters for RS
        rs_x_data = np.repeat(nM, rs_log.shape[1])
        rs_y_data = rs_log.flatten()
        rs_r_squared, rs_params = fit_and_calculate_r2(rs_x_data, rs_y_data)
        rs_log_mean = rs_log.mean(axis=1)
        rs_log_std = rs_log.std(axis=1, ddof=1)

        # Prepare data and calculate R-squared and parameters for IAC
        iac_x_data = np.repeat(nM, iac_log.shape[1])
        iac_y_data = iac_log.flatten()
        iac_r_squared, iac_params = fit_and_calculate_r2(iac_x_data, iac_y_data)
        iac_log_mean = iac_log.mean(axis=1)
        iac_log_std = iac_log.std(axis=1, ddof=1)

        # Prepare data and calculate R-squared and parameters for TS
        ts_x_data = np.repeat(nM, ts_log.shape[1])
        ts_y_data = ts_log.flatten()
        ts_r_squared, ts_params = fit_and_calculate_r2(ts_x_data, ts_y_data)
        ts_log_mean = ts_log.mean(axis=1)
        ts_log_std = ts_log.std(axis=1, ddof=1)

        st.write("Combined 4PL Curves")
        plot_combined_4pl_curves(nM, rs_log_mean, rs_params, rs_log_std, 
                                iac_log_mean, iac_params, iac_log_std, 
                                ts_log_mean, ts_params, ts_log_std)
        
        all_data = {
            "Type": ["RS","IAC","TS"],
            "R-squared": [rs_r_squared, iac_r_squared, ts_r_squared],
            "a": [rs_params[0], iac_params[0], ts_params[0]],
            "b": [rs_params[1], iac_params[1], ts_params[1]],
            "c": [rs_params[2], iac_params[2], ts_params[2]],
            "d": [rs_params[3], iac_params[3], ts_params[3]],
        }

        # Create a DataFrame
        df = pd.DataFrame(all_data)

        # Display the DataFrame
        st.write("Model Parameters and R-squared Values")
        st.dataframe(df)

        # Calculate and display the ratio of parameters for IAC
        percent_RP = np.round((rs_params[2] / iac_params[2]) * 100, 2)
        ratio_b_parameter = np.round(iac_params[1] / rs_params[1], 2)
        ratio_a_parameter = np.round(iac_params[0] / rs_params[0], 2)
        ratio_d_parameter = np.round(iac_params[3] / rs_params[3], 2)

        iac_data ={
            "%RP": [percent_RP],
            "Ratio of B parameter": [ratio_b_parameter],
            "Ratio of A parameter": [ratio_a_parameter],
            "Ratio of D parameter": [ratio_d_parameter],
        }

        # Create a DataFrame
        iac_data_df = pd.DataFrame(iac_data)

        # Display the DataFrame
        st.write("Ratio of IAC Parameters")
        st.dataframe(iac_data_df)
       
        iac_within_spec = True
        if not (50 <= percent_RP <= 150):
            st.write("IAC % RP is out of specifications.")
            iac_within_spec = False
        if not (0.5 <= ratio_b_parameter <= 2.0):
            st.write("IAC Ratio of B parameter is out of specifications.")
            iac_within_spec = False
        if not (0.8 <= ratio_a_parameter <= 1.25):
            st.write("IAC Ratio of A parameter is out of specifications.")
            iac_within_spec = False
        if not (0.8 <= ratio_d_parameter <= 1.25):
            st.write("IAC Ratio of D parameter is out of specifications.")
            iac_within_spec = False

        if iac_within_spec:
            st.write("IAC parameters are within specifications.")
        else:
            st.write("IAC parameters are out of specifications.")

        division_factors = {
            '40%_1': 0.4,
            '70%_1': 0.7,
            '100%_1': 1,
            '130%_1': 1.3,
            '160%_1': 1.6,
            '40%_2': 0.4,
            '70%_2': 0.7,
            '100%_2': 1,
            '130%_2': 1.3,
            '160%_2': 1.6,
            '40%_3': 0.4,
            '70%_3': 0.7,
            '100%_3': 1,
            '130%_3': 1.3,
            '160%_3': 1.6,
            # Add more entries as needed for each unique page identifier
        }

        # Generate a unique key for the current page
        division_factor = division_factors.get(page, 1)

        # Calculate and display the ratio of parameters for the test sample
        ratio_b_parameter_ts = round(ts_params[1] / rs_params[1], 2)
        ratio_a_parameter_ts = round(ts_params[0] / rs_params[0], 2)
        ratio_d_parameter_ts = round(ts_params[3] / rs_params[3], 2)
        percent_recovery = round((rs_params[2] / ts_params[2] / division_factor) * 100, 2)

        ts_data ={
            "% Recovery": [percent_recovery],
            "Ratio of B parameter": [ratio_b_parameter_ts],
            "Ratio of A parameter": [ratio_a_parameter_ts],
            "Ratio of D parameter": [ratio_d_parameter_ts],
        }

        # Create a DataFrame
        ts_data_df = pd.DataFrame(ts_data)

        # Display the DataFrame
        st.write("Ratio of TS Parameters")
        st.dataframe(ts_data_df)

        # Check if Test Sample parameters are within specifications
        ts_within_spec = True
        if not (80 <= percent_recovery <= 120):
            st.write("Test Sample % Recovery is out of specifications.")
            ts_within_spec = False
        if not (0.5 <= ratio_b_parameter_ts <= 2.0):
            st.write("Test Sample Ratio of B parameter is out of specifications.")
            ts_within_spec = False
        if not (0.8 <= ratio_a_parameter_ts <= 1.25):
            st.write("Test Sample Ratio of A parameter is out of specifications.")
            ts_within_spec = False
        if not (0.8 <= ratio_d_parameter_ts <= 1.25):
            st.write("Test Sample Ratio of D parameter is out of specifications.")
            ts_within_spec = False

        if ts_within_spec:
            st.write("Test Sample parameters are within specifications.")
        else:
            st.write("Test Sample parameters are out of specifications.")

        authenticator.logout("Logout", "sidebar")

