'''
This script is a Python implementation of the proof-of-concept data model for personality trait prediction by Dave Smith, from his 
article "Weapons of Micro Destruction: How Our 'Likes' Hijacked Democracy" (https://towardsdatascience.com/weapons-of-micro-destruction-how-our-likes-hijacked-democracy-c9ab6fcd3d02), itself an attempt to reverse-engineer a POC of CA's model. It also 
serves as an exercise for me in learning and understanding more about linear regression, especially L1 regularization with LASSO.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''
   
    frame: pd.DataFrame = pd.read_csv("Data/WeaponsOfMicroDestructionRawData.csv");
    return frame


# =====================================================================


def get_features_and_labels(frame):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0
    # or we can use scikit-learn to calculate missing values below
    #frame[frame.isnull()] = 0.0

    # Convert values to floats
    frame = frame.drop(["Person"], axis=1)
    arr = np.array(frame, dtype=np.float)

    # Normalize the entire data set
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    #arr = StandardScaler().fit_transform(arr)

    # Use the last column as the target value
    #X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    X, y = arr[:, 1:], arr[:, 0]
    
    
    X_train, y_train = X[:5, :], y[:5]
    X_test, y_test = X[5:, :], y[5:]
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))   
    
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    y_train_mean = np.average(y_train)
    y_train_std = np.std(y_train)
    y_test_mean = np.average(y_test)
    y_test_std = np.std(y_test)
    y_train = (y_train - y_train_mean)/y_train_std
    y_test = (y_test - y_train_mean)/y_train_std #NOTE: this is what Smith did in his Excel calculations, but I do not understand the reasoning behind it.
    # Why would you apply the mean & std from the training data instead of the test data when standardizing the test data labels?
    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================


def evaluate_learner(X_train, X_test, y_train, y_test, alpha):
    '''

    Returns a sequence of tuples containing:
        (title, expected values, actual values)
    for each learner.
    '''

    # Use a LASSO model for regression
    from sklearn.linear_model import Lasso

    # Train using an alpha/lambda value I customize and random (rather than cyclic) updating of coefficients per iteration
    lasso = Lasso(alpha, tol=0.001)
    lasso.fit(X_train, y_train)
    print("Lasso coefficients at convergence:")
    print(lasso.coef_)
    y_pred = lasso.predict(X_test)
    print("Predicted vs. actual openness scores:")
    print(y_pred, y_test)
    mse = np.sum((y_test - y_pred)**2)/2
    r_2 = lasso.score(X_test, y_test)
    yield 'Lasso Model w/ Alpha = {0} ($R^2={1:.3f}$, MSE={2})'.format(alpha, r_2, mse), y_test, y_pred


if __name__ == '__main__':
    # Download the data set from URL
    print("Downloading data from file")
    frame = download_data()

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple regression learners on the data
    print("Evaluating regression learners")
    alphaList = [0, 0.01, 0.1, 1.0]
    results = []
    for i in alphaList:
        results.append(list(evaluate_learner(X_train, X_test, y_train, y_test, i)))
    print(results)
    
