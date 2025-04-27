from statsmodels.tsa.holtwinters import ExponentialSmoothing
from model_pipeline import *

def holt_winters_back_testing_1(df, seasonal_periods):
    """
    Implement back testing 1 for Holt-Winters model

    Parameters:
        df (pd.DataFrame): Dataframe.
        seasonal_periods (int): Number of periods to use for seasonality.
    """

    # Create DataFrame to contain predictions
    df_back_test_1 = pd.DataFrame()

    # Get train and test data
    train, test = get_train_test_back_testing_1(df)

    # Split test to X and y
    X_train, y_train = get_x_y(train)
    X_test, y_test = get_x_y(test)

    # Build and fit Holt-Winters model
    model = ExponentialSmoothing(
        y_train,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    ).fit()

    # Get the predictions and actuals value
    predictions, actuals = get_predictions_actual_back_test_1(model, X_test, y_test)

    # Save it to DataFrame
    for i in range(len(predictions)):
        df_back_test_1 = df_back_test_1._append({'Prediction':predictions.iloc[i],
                                                 'Actual':actuals.iloc[i]},
                                                 ignore_index=True)

    # Evaluate Holt-Winters model when using back testing 1
    print("Evaluate Holt-Winters model when using back testing 1:")
    evaluate_model(model, actuals, predictions)

    # Show the predictions and actuals plot
    show_predict_actual_plot(df_back_test_1, df,"Holt-Winters Back Testing 1", TEST_SIZE)


def holt_winters_back_testing_2(df, seasonal_periods):
    """
    Implement back testing 2 for Holt-Winters model

    Parameters:
        df (pd.DataFrame): Dataframe.
        seasonal_periods (int): Number of periods to use for seasonality.
    """

    # Create DataFrame to contain predictions
    df_back_test_2 = pd.DataFrame()

    for i in range(len(df) - TEST_SIZE):
        # Calculate the num days ahead
        num_days_ahead = TEST_SIZE - i
        if num_days_ahead <= 0:
            break

        # Get train data
        train = get_train_back_testing_2(df, i)

        # Get test data
        test = get_test_back_testing_2(df, i)

        # Split train and test to X and y
        X_train, y_train = get_x_y(train)
        X_test, y_test = get_x_y(test)

        # Build and fit Holt-Winters model
        model = ExponentialSmoothing(
            y_train,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods
        ).fit()

        # Get the prediction and actual value
        prediction, actual = get_predictions_actual_back_test_2(model, X_test, y_test)

        # Save it to DataFrame
        df_back_test_2 = df_back_test_2._append({'Prediction':prediction,
                                                 'Actual':actual},
                                                 ignore_index=True)

    # Evaluate Holt-Winters model when using back testing 2
    print("Evaluate Holt-Winters model when using back testing 2:")
    evaluate_model(model, df_back_test_2['Actual'], df_back_test_2['Prediction'])

    # Show the predictions and actuals plot
    show_predict_actual_plot(df_back_test_2, df,"Holt-Winters Back Testing 2", TEST_SIZE)


holt_winters_back_testing_1(df, seasonal_periods=35)
holt_winters_back_testing_2(df, seasonal_periods=35)
