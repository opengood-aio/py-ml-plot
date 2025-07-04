# Python Machine Learning (ML) Plot

[![Build](https://github.com/opengood-aio/py-ml-plot/workflows/build/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions?query=workflow%3Abuild)
[![Release](https://github.com/opengood-aio/py-ml-plot/workflows/release/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions?query=workflow%3Arelease)
[![CodeQL](https://github.com/opengood-aio/py-ml-plot/actions/workflows/codeql.yml/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions/workflows/codeql.yml)
[![Codecov](https://codecov.io/gh/opengood-aio/py-ml-plot/graph/badge.svg?token=WX6Er5S6Vj)](https://codecov.io/gh/opengood-aio/py-ml-plot)
[![Release Version](https://img.shields.io/github/release/opengood-aio/py-ml-plot.svg)](https://github.com/opengood-aio/py-ml-plot/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/opengood.py-ml-plot)](https://pypi.org/project/opengood.py-ml-plot/)
![Python](https://img.shields.io/pypi/pyversions/opengood.py-ml-plot)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/opengood-aio/py-ml-plot/master/LICENSE)

Modules containing reusable functions for machine learning visualization
plotting

## Compatibility

* Python 3.13 or later

## Setup

### Add Dependency

```bash
python3 -m pip install opengood.py-ml-plot
```

**Note:** See *Release* version badge above for the latest version.

---

## Features

### Classification Model Plotting

#### Set Up and Display 2-D Classification Plot

Set up a 2-D classification model plot then display its result visualization.

**Notes:**

* The example below uses a dataset to train a logistic regression model then
  display the plot for the training set
* For feature scaling, if required, implement the feature scaling logic in the
  `feature_scaling` lambda
* For predictions, implement the prediction logic in the `predict` lambda

```python
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.opengood.py_ml_plot import setup_classification_plot

resource_path = os.path.join(os.path.dirname(__file__), "../resources", "data.csv")
dataset = pd.read_csv(resource_path)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

setup_classification_plot(
    x=x_train,
    y=y_train,
    cmap=ListedColormap(("salmon", "dodgerblue")),
    title="Logistic Regression",
    x_label="Age",
    y_label="Estimated Salary",
    feature_scale=lambda x_set, y_set: (
        sc.inverse_transform(x_set), y_set
    ),
    predict=lambda x1, x2: (
        classifier.predict(
            sc.transform(
                np.array([x1.ravel(), x2.ravel()]).T)
        ).reshape(x1.shape)
    ),
)

plt.show()
```

##### Output

![Setup Classification Plot Visualization](https://raw.githubusercontent.com/opengood-aio/py-ml-plot/main/docs/images/setup-classification-plot-visualization.png)

---

##### Visualization Logic

###### Feature Scale Lambda Logic

`feature_scale` lambda implementation logic for function
`setup_classification_plot` is as follows:

* Inverse feature scaling is invoked via a featuring scaling object, such as
  the `StandardScalar` object `sc` created earlier for feature scaling
* `x_set` and `y_set` are assigned non-feature scaled values of the matrix of
  features and the dependent variable
  * `x_set` values are inverted from their feature-scaled values in `x`
  * `y_set` values are not inverted and taken directly from `y`

###### Predict Lambda Logic

`predict` lambda implementation logic for function `setup_classification_plot`
is as follows:

* Classifier object `classifier` method `predict` is invoked
* Since the values of the reshaped 2D array are not feature scaled, the
  values are feature scaled via the `transform` method on the `sc`
  object
    * This method call is not required for models that do not require feature
      scaling
* `ravel` function from the NumPy library is used to flatten a
  multidimensional array into a one-dimensional array
  * `x1` and `x2` are flatten into a 1D array via the `ravel` function
  * They are then combined via the `array` function from the NumPy
  library into a 2D array
  * The result is then reshaped via the `reshape` function to match
  the shape of `x1`

###### Setup Plot Visualization Logic

Visualization implementation logic for function `setup_classification_plot` is
as follows:

* `ListedColormap` class from the Matplotlib library creates an object that
  generates a colormap visual from a list of colors
* If the `feature_scale` lambda is defined, `x_set` and `y_set` are
  assigned non-feature scaled values of the matrix of features and the dependent
  variable from the sets using a feature scaling object, such as the
  `StandardScalar` object created earlier for feature scaling
    * `x_set` values are inverted from their feature-scaled values in `x`
    * `y_set` values are not inverted and taken directly from `y`
* If the `feature_scale` lambda is not defined, `x_set` and `y_set` are
  assigned the values of `x` and `y`, respectively
* `meshgrid` function from the NumPy library returns a tuple of coordinate
  matrices from coordinate vectors
    * Two sets of matrices (`x1` and `x2`) are returned with coordinate vectors
    * `x1`
        * `arange` function is called with a defined start and stop interval
            * `x_set[:, 0]` returns all the rows for feature `x1`
            * `start` parameter
                * Start of an interval
                * `x_set[:, 0].min()` returns the minimum value for feature `x1`
                * Value of `10` is subtracted for padding
            * `stop` parameter
                * End of an interval
                * `x_set[:, 0].max()` returns the maximum value for feature `x1`
                * Value of `10` is added for padding
            * `step` parameter
                * Spacing between values
                * Value of `0.25` is added for spacing
    * `x2`
        * `arange` function is called with a defined start and stop interval
            * `x_set[:, 1]` returns all the rows for feature `x2`
            * `start` parameter
                * Start of an interval
                * `x_set[:, 1].min()` returns the minimum value for feature `x2`
                * Value of `1000` is subtracted for padding
                * Value of `1000` is used instead of `10` due to the difference
                  in scaling for feature `x2` vs. feature `x1`
            * `stop` parameter
                * End of interval
                * `x_set[:, 1].max()` returns the maximum value for feature `x2`
                * Value of `1000` is added for padding
            * `step` parameter
                * Spacing between values
                * Value of `0.25` is added for spacing
* The prediction logic implemented in the `preodict` lambda is executed, and
  the result is assigned to `y_pred`, containing the predictions
* `contourf` function from the Matplotlib library is used for creating filled
  contour plots
    * It visualizes 3D data in 2D by drawing filled contours representing
      constant z-values (heights) on an x-y plane
    * These plots are useful for displaying data like temperature distributions,
      terrain elevations, or any scalar field where the magnitude varies over 2
      dimensions
    * The most basic use case of `contourf` involves providing a 2D array
      representing the z-values
    * Matplotlib automatically determines the x and y coordinates based on the
      array's indices
    * `X` and `Y` parameters
        * The coordinates of the values in `Z`
        * `X` and `Y` must both be 2D arrays with the same shape as `Z`
        * `x1` is used for `X` containing `x1` values
        * `x2` is used for `Y` containing `x2` values
    * `Z` parameter
        * The height values over which the contour is drawn
        * `ravel` function from the NumPy library is used to flatten a
          multidimensional array into a one-dimensional array
            * `x1` and `x2` are flatten into a 1D array via the `ravel` function
            * They are then combined via the `array` function from the NumPy
              library into a 2D array
            * The result is then reshaped via the `reshape` function to match
              the shape of `x1`
        * Since the values of the reshaped 2D array are not feature scaled, the
          values are feature scaled via the `transform` method on the `sc`
          object
    * `alpha` parameter
        * The alpha blending value, between `0` (transparent) and `1` (opaque)
        * Value of `0.75` is used to make the blending mostly opaque
    * `cmap` parameter
        * The `Colormap` object instance or registered colormap name used to map
          scalar data to colors
        * `salmon` and `dodgerblue` are used for a `ListedColormap` object
            * `salmon` = 0 or negative classifier
            * `dodgerblue` = 1 or positive classifier
* `xlim` function from the Matplotlib library is used to get or set the x-axis
  limits of the current axes
    * `min()` and `max()` for `x1` are used for the limits
* `ylim` function from the Matplotlib library is used to get or set the y-axis
  limits of the current axes
    * `min()` and `max()` for `x2` are used for the limits
* The values from `y_set` are iterated over in a for-in loop
    * `unique` function from the NumPy library returns sorted, unique elements
      of an array
        * Values of `y_set` are made unique and sorted
    * Iterator variable `i` represents the current row of iteration
    * Iterator variable `j` represents the classification value for the
      dependent variable
        * `0` negative classifier
        * `1` positive classifier
* `scatter` method from the Matplotlib library creates a scatter plot of data
  points with the shaded contour showing the classification for the dependent
  variable
    * x-axis uses values from `x_set` where `y_set` value = 0 (negative
      classifier)
    * y-axis uses values from `x_set` where `y_set` value = 1 (positive
      classifier)
    * `c` parameter
        * The marker colors
        * Uses the `ListedColormap` with the classification colors for the
          current row at index `i`
    * `label` parameter
        * Sets the label
        * Values
            * `0` negative classifier
            * `1` positive classifier

---

# Development

## Python Virtual Environment

Create Python virtual environment:

```bash
cd ~/workspace/opengood-aio/py-ml-plot/.venv
python3 -m venv ~/workspace/opengood-aio/py-ml-plot/.venv
source .venv/bin/activate
```

## Install Packages

```bash
python3 -m pip install matplotlib
python3 -m pip install numpy
python3 -m pip install pandas
python3 -m pip install scikit-learn
```

## Create Requirements File

```bash
pip freeze > requirements.txt
```

## Run Tests

```bash
python -m pytest tests/
```

