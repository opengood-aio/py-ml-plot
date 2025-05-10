# Python Machine Learning (ML) Plot

[![Build](https://github.com/opengood-aio/py-ml-plot/workflows/build/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions?query=workflow%3Abuild)
[![Release](https://github.com/opengood-aio/py-ml-plot/workflows/release/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions?query=workflow%3Arelease)
[![CodeQL](https://github.com/opengood-aio/py-ml-plot/actions/workflows/codeql.yml/badge.svg)](https://github.com/opengood-aio/py-ml-plot/actions/workflows/codeql.yml)
[![Codecov](https://codecov.io/gh/opengood-aio/py-ml-plot/graph/badge.svg?token=WX6Er5S6Vj)](https://codecov.io/gh/opengood-aio/py-ml-plot)
[![Release Version](https://img.shields.io/github/release/opengood-aio/py-ml-plot.svg)](https://github.com/opengood-aio/py-ml-plot/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/py-ml-plot)](https://pypi.org/project/py-ml-plot/)
![Python](https://img.shields.io/pypi/pyversions/py-ml-plot)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/opengood-aio/py-ml-plot/master/LICENSE)

Modules containing reusable functions for machine learning visualization
plotting

# Setup

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

