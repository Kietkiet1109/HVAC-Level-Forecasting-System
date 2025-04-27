import pandas as pd
import warnings
from plot_draw import plot_all_features_heatmap

# Global variables
warnings.filterwarnings("ignore")
CSV_PATH = "HVAC_Level_Preprocessed.csv"

def read_csv(path):
    """
    Read the CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): corresponding Dataframe.
    """

    # Reading CSV data
    df = pd.read_csv(path)

    return df


def evaluate_data(df):
    """
    Evaluate DataFrame.

    Parameters:
        df (pd.DataFrame): Raw Dataframe.
    """

    # Config data display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Print the DataFrame
    print(df)

    # Print the DataFrame summary
    print(df.describe())
    print("\n")
    print(df.info())

    # Delete the Date column out of DataFrame
    del df['Date']

    # Plot the heatmap for all features
    plot_all_features_heatmap(df)


df = read_csv(CSV_PATH)
evaluate_data(df)
