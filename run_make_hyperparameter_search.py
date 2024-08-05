########
import numpy as np


def load_data():

    import pandas as pd

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return x, y


def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
    )
    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2


def report(estimator, mse, mae, r2):

    print(estimator, ":", sep="")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")


def save_best_estimator(estimator):

    import os
    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)


def load_best_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def train_estimator(alpha=0.5, l1_ratio=0.5, verbose=1):

    from sklearn.linear_model import ElasticNet

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
    estimator.fit(x_train, y_train)
    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))

    if verbose > 0:
        report(estimator, mse, mae, r2)

    best_estimator = load_best_estimator()
    if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(
        x_test, y_test
    ):
        best_estimator = estimator

    save_best_estimator(best_estimator)


def check_estimator():

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)
    estimator = load_best_estimator()
    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    report(estimator, mse, mae, r2)


#
# Debe coincidir con el mejor modelo encontrado en la celdas anteriores
#
check_estimator()


def make_hyperparameters_search(alphas, l1_ratios):

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            train_estimator(alpha=alpha, l1_ratio=l1_ratio, verbose=0)


alphas = np.linspace(0.0001, 0.5, 10)
l1_ratios = np.linspace(0.0001, 0.5, 10)
make_hyperparameters_search(alphas, l1_ratios)
check_estimator()


######
