import os

import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from opengood.py_ml_plot import display_classification_plot


def test_display_classification_plot_with_shaded_regions():
    """Should display a classification plot with classified regions shaded"""

    resource_path = os.path.join(os.path.dirname(__file__), "resources", "data.csv")
    dataset = pd.read_csv(resource_path)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, _, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)

    result = display_classification_plot(
        x_train,
        y_train,
        sc,
        classifier,
        ListedColormap(("salmon", "dodgerblue")),
        "Logistic Regression (Training Set)",
        "Age",
        "Estimated Salary",
    )
    assert result is True
