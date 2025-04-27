import pandas as pd
import numpy as np
import joblib
import warnings
import statsmodels.api as sm

# Define constants
warnings.filterwarnings("ignore")
CSV_PATH = "HVAC_Level_Preprocessed.csv"
MODEL_PATH = "OLS_model.pkl"
SCALER_PATH = "standard_scaler.pkl"
PRED_PATH = "HVAC_Level_Predictions.csv"
TARGET = "HVAC_Level"
PREDICT_SIZE = 6

def read_csv(path):
    """
    Read the CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Dataframe.
    """

    # Reading raw data
    df = pd.read_csv(path)

    return df


def get_x(df, scaler):
    """
    Get the X from DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        X (pd.DataFrame): Transformation Dataframe with selected columns.
    """

    # Get X DataFrame with top 10 selected columns.
    # Top 10 features take from feature_selection()
    X = df[['A', 'CC', 'H', 'I', 'Z', 'E', 'F', 'M', 'BB', 'HVAC_Level_t-1']]

    # Scale the data
    X_scaled = scaler.fit_transform(X)

    # Add constant to X for intercept
    X = sm.add_constant(X_scaled, has_constant='add')

    return X


def get_date_range(df):
    """
    Get the prediction date range.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        date_range (NumPy Array): Dates range of predictions
    """

    # Get the last day in the DataFrame
    last_day = df['Date'].max()

    # Get the date range for predictions
    date_range = (pd.date_range(start=last_day, periods=PREDICT_SIZE + 1, freq='D')
                  .to_numpy())

    # Remove the first day (the last day in DataFrame) in the date range
    # We will make predictions for the next 6 days after last day
    date_range = date_range[1:]

    return date_range


def get_predictions(df, model, date_range):
    """
    Get the predictions DataFrame from OLS model.

    Parameters:
        df (pd.DataFrame): DataFrame with selected columns.
        model: OLS model.
        date_range (NumPy Array): Dates range of predictions

    Returns:
        df_predictions (pd.DataFrame): Dataframe store predictions for next 6 days.
    """

    # Create a DataFrame to store grade for next 6 days
    df_predictions = pd.DataFrame()

    for i in range(PREDICT_SIZE):
        # Make prediction for each day
        predictions = model.predict(df)

        # Save it to DataFrame
        df_predictions = df_predictions._append({'Date': date_range[i],
                                                 TARGET: predictions[i].round(3)},
                                                 ignore_index=True)

    # Print out the predictions DataFrame
    print(df_predictions)

    return df_predictions


# Read the CSV file.
df = read_csv(CSV_PATH)

# Load the Standard Scaler
scaler = joblib.load(SCALER_PATH)

# Get X DataFrame
X = get_x(df, scaler)

# Load the model
model = joblib.load(MODEL_PATH)

# Get the date range for predictions
date_range = get_date_range(df)

# Get predictions DataFrame
df_predictions = get_predictions(X, model, date_range)

# Export predictions to .csv file
df_predictions.to_csv(PRED_PATH, index=False)
