import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from scipy import stats
from plot_draw import (plot_time_trending, plot_best_feature_decomposition,
                       plot_selected_features_heatmap, show_predict_actual_plot)

# Global variables
warnings.filterwarnings("ignore")
RAW_CSV_PATH = "HVAC_Level.csv"
PREPROCESSED_CSV_PATH = "HVAC_Level_Preprocessed.csv"
TARGET = "HVAC_Level"
THRESHOLD = 3
N_ROWS = 3
N_COLS = 3

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


def convert_date(df):
    """
    Convert date columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with date converted.
    """

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    return df


def implement_back_shifting(df):
    """
    Create 2 new columns for back shifting 1 row and 2 rows.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with 2 new back shifting columns.
    """

    # Back shifting 1 row
    df['HVAC_Level_t-1'] = df['HVAC_Level'].shift(1)

    # Back shifting 2 row
    df['HVAC_Level_t-2'] = df['HVAC_Level'].shift(2)

    # Drop any NA value in 2 new columns
    df.dropna(inplace=True)

    # Create EMA over 3 observations
    df['EMA_3'] = df['HVAC_Level'].ewm(span=3, adjust=False).mean()

    return df


def feature_selection(df):
    """
    Get the top 10 features based on their correlation with HVAC_Level.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        selected_features (Numpy Array): Array with selected columns.
    """

    # Calculate the correlation matrix.
    corr = df.corr()

    # Sorted the correlations based on their correlation
    sorted_corr = corr[[TARGET]].sort_values(by=TARGET, ascending=False)

    # Take the top 10 features
    # Start from 1 to skip the best correlation with HVAC_Level
    # HVAC_Level will have the best correlation with itself
    selected_features = sorted_corr.index[1:11].to_numpy()

    return selected_features


def export_csv(df, path):
    """
    Export DataFrame to csv file.

    Parameters:
        df (pd.DataFrame): Dataframe.
        path (str): Path for new csv file.
    """

    df.to_csv(path, index=False)


df = read_csv(RAW_CSV_PATH)
converted_df = convert_date(df)
final_df = implement_back_shifting(converted_df)
selected_features = feature_selection(df)
plot_time_trending(converted_df, selected_features)
plot_best_feature_decomposition(converted_df)
plot_selected_features_heatmap(df[selected_features])
export_csv(final_df, PREPROCESSED_CSV_PATH)
