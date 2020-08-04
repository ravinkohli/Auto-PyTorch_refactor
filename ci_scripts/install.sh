#!/usr/bin/env bash

pip install pep8 codecov mypy flake8 pytest-cov flake8-import-order
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install .[all]
