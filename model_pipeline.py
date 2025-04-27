import joblib
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from data_preprocess import *

# Global variables
TEST_SIZE   = 10
SCALER_PATH = "standard_scaler.pkl"
SCALER = StandardScaler()

def get_train_test_back_testing_1(df):
    """
    Get train and test data for back testing 1

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        train (pd.DataFrame): train data
        test (pd.DataFrame): test data
    """

    # Split data into train and test data.
    train = df.iloc[:len(df)-TEST_SIZE]
    test = df.iloc[len(df)-TEST_SIZE:]

    return train, test


def get_train_back_testing_2(df, index):
    """
    Get train data for back testing 2

    Parameters:
        df (pd.DataFrame): Dataframe.
        index (int): index of current iteration

    Returns:
        train (pd.DataFrame): train data
    """

    # Initialize start index
    start_index = 0

    # Calculate end index
    end_index = index + (len(df) - TEST_SIZE)

    # Get the train data
    train = df.iloc[start_index:end_index]

    return train


def get_test_back_testing_2(df, index):
    """
    Get test data for back testing 2

    Parameters:
        df (pd.DataFrame): Dataframe.
        index (int): index of current iteration

    Returns:
        test (pd.DataFrame): test data
    """

    # Initialize start index
    start_index = index + (len(df) - TEST_SIZE)

    # Calculate end index
    end_index = index + len(df)

    # Get the test data
    test = df.iloc[start_index:end_index]

    return test


def scale_data(X, scaler):
    """
    Scale data using passing Scaler.

    Parameters:
        X (pd.DataFrame): Dataframe needed to scale.
        scaler (Scaler): Scaler that been passed from parameter.

    Returns:
        X_scaled (NumPy ndarray): NumPy array containing the transformed data.
    """

    # Fit the scaler to X
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def drop_outliers(X, y):
    """
    Drop outliers in X and y.

    Parameters:
        X (pd.DataFrame): Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.

    Returns:
        X (pd.DataFrame): X without outliers.
        y (pd.DataFrame): y without outliers.
    """

    # Calculate Z score for each value in X
    z = np.abs(stats.zscore(X))

    # Find any value have z-score greater than THRESHOLD
    outliers = np.where(z > THRESHOLD)

    # Remove outliers from X and y
    X.drop(outliers[0], inplace=True)
    y.drop(outliers[0], inplace=True)

    # Reset index for both X and y after dropping
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y


def get_x_y(df):
    """
    Get the X and y from DataFrame.

    Parameters:
        df (pd.DataFrame): Dataframe.

    Returns:
        X (pd.DataFrame): Transformation Dataframe with selected columns.
        y (pd.DataFrame): Dataframe with target column.
    """

    # Get X DataFrame without Date and EMA_3 column.
    X = df.copy()
    del X['Date'], X['EMA_3']

    # Get top 10 features
    selected_features = feature_selection(X)

    # Get X DataFrame with selected columns.
    X = df[selected_features]

    # Get y DataFrame with the target column.
    y = df[TARGET]

    # Drop outliers for X and y
    X, y = drop_outliers(X, y)

    # Scale X data
    X_scaled = scale_data(X, SCALER)

    # Add constant to X for intercept
    X = sm.add_constant(X_scaled, has_constant='add')

    return X, y


def get_predictions_actual_back_test_1(model, X_test, y_test):
    """
    Get the predictions from model and actuals value for back test 1

    Parameters:
        model: Passing model.
        X_test (NumPy ndarray): Data of X test.
        y_test (NumPy ndarray): Data of y test.

    Returns:
        predictions (NumPy ndarray): Predicted target data.
        actuals (NumPy ndarray): Actual target data.
    """

    # Get the model type
    model_type = model.__class__.__name__

    # Check the model type
    if model_type == 'RegressionResultsWrapper':
        # Make the predictions by Linear Regression Model.
        predictions = model.predict(X_test)

    else:
        # Make the predictions by Holt-Winter or ARIMA model.
        predictions = model.forecast(TEST_SIZE)

    # Get the actuals value
    actuals = y_test

    return predictions, actuals


def get_predictions_actual_back_test_2(model, X_test, y_test):
    """
    Get the predictions from model and actuals value for back test 2

    Parameters:
        model: Passing model.
        X_test (NumPy ndarray): Data of X test.
        y_test (NumPy ndarray): Data of y test.

    Returns:
        predictions (NumPy ndarray): Predicted target data.
        actuals (NumPy ndarray): Actual target data.
    """

    # Get the model type
    model_type = model.__class__.__name__

    # Check the model type
    if model_type == 'RegressionResultsWrapper':
        # Make the predictions by Linear Regression Model.
        predictions = model.predict(X_test)[0]

    else:
        # Make the predictions by Holt-Winter or ARIMA model.
        predictions = model.forecast(1).iloc[0]

    # Get the actuals value
    actuals = y_test.iloc[0]

    return predictions, actuals


def evaluate_model(model, y_test, predictions):
    """
    Evaluate appropriate model.

    Parameters:
        model: Passing model.
        y_test (pd.DataFrame): Dataframe with target column.
        predictions (NumPy ndarray): NumPy array with predicted values.
    """

    # Print the model summary
    print(model.summary())

    # Calculate the rmse
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

    # Get the model type
    model_type = model.__class__.__name__

    # Check the model type
    if model_type == 'RegressionResultsWrapper':
        print('Root Mean Squared Error of OLS Model:',rmse)

    elif model_type == 'HoltWintersResultsWrapper':
        print('Root Mean Squared Error of Holt-Winters Model:', rmse)

    else:
        print('Root Mean Squared Error of ARIMA Model:', rmse)


def export_model(model, path):
    """
    Export model and scaler as .pkl file.

    Parameters:
        model: passing model.
        path (str): Path to save model.
    """

    joblib.dump(model, path)
    joblib.dump(SCALER, SCALER_PATH)
