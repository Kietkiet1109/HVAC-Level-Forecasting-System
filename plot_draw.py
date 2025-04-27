import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

# Global variables
TARGET = 'HVAC_Level'

def plot_time_trending(df, selected_features):
    """
    Create a plot for the time trending of the selected features.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        selected_features (Numpy Array): Array with selected columns.
    """

    # Create a figure and axes object.
    # Set up plot size.
    fig, ax = plt.subplots(figsize=(10, 10))

    # Loop through each feature in the Array.
    for col in selected_features:
        # Plot each feature.
        ax.plot(df['Date'], df[col], label=col)

    # Title of the plot.
    ax.set_title('Feature Trends Over Time', fontsize=16, fontweight='bold')

    # Label axis in the plot.
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)

    # Show the legend on the plot.
    ax.legend()

    # Draw grid lines at each 2 months.
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.grid(True, linewidth=0.5, color='gray')

    # Set up x-tick.
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.show()


def plot_best_feature_decomposition(df):
    """
    Create a plot for best feature decomposition in the selected features.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
    """

    # Convert Date to index
    df = df.set_index('Date')

    # Perform decomposition using multiplicative decomposition.
    t_series = seasonal_decompose(df['A'], model='multiplicative',
                                  period=6, extrapolate_trend="freq")

    # Create a figure object.
    # Set up plot size.
    fig = t_series.plot()
    fig.set_size_inches(10, 10)

    # Loop for each ax in figure
    for ax in fig.axes:
        # Set x-axis range
        # ax.set_xlim(start, end)

        # Draw grid lines at each 2 months.
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.grid(True, linewidth=0.5, color='gray')

        # Set up x-tick.
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.show()


def plot_all_features_heatmap(features):
    """
    Create a heatmap between all feature and target.

    Parameters:
        features (pd.DataFrame): DataFrame contains all features.
    """

    # Set up heatmap size.
    plt.figure(figsize=(10, 10))

    # Calculate the correlation matrix.
    corr = features.corr()

    # Sort the correlations in descending order
    sorted_corr = corr[[TARGET]].sort_values(by=TARGET, ascending=False)

    # Plot a heatmap.
    sns.heatmap(sorted_corr, annot=True, cmap="coolwarm", fmt=".2f")

    # Title of heatmap
    plt.title(f"Heatmap comparison between all features and {TARGET}",
              fontsize=16, fontweight='bold'
              )

    plt.show()


def plot_selected_features_heatmap(features):
    """
    Create a heatmap between 10 selected feature and target.

    Parameters:
        features (pd.DataFrame): DataFrame contains 10 selected features.
    """

    # Set up heatmap size.
    plt.figure(figsize=(10, 10))

    # Calculate the correlation matrix.
    corr = features.corr()

    # Plot a heatmap.
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    # Title of heatmap
    plt.title(f"Heatmap comparison between 10 selected features and {TARGET}",
              fontsize=16, fontweight='bold'
              )

    plt.show()


def show_predict_actual_plot(df_back_test, df, name, num_day_predict):
    """
    Create a plot to compare the predictions and actuals values.

    Parameters:
        df_back_test (pd.DataFrame): DataFrame contains predictions and actuals values.
        df (pd.DataFrame): Original DataFrame.
        name (str): Name of model.
        num_day_predict (int): Number of days to predict.
    """

    # Convert Date to index
    df = df.set_index('Date')

    # Take the final rows
    final_rows = df.tail(num_day_predict)

    # Plot the predictions line
    plt.plot(final_rows.index, df_back_test['Prediction'],
             label='Prediction', color='orange')

    # Plot the actuals line
    plt.plot(final_rows.index, df_back_test['Actual'],
             label='Actual', color='blue')

    # Show the legend on the plot.
    plt.legend()

    # Title of the plot.
    plt.title(f"{num_day_predict} Days {name} Predictions")

    # Set up x-tick.
    plt.xticks(rotation=70)

    plt.tight_layout()

    plt.show()
