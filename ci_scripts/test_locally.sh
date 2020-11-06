# Activate virtual environment
# Note: Make sure to install flake8 and mypy in the virtual environment
# source env/bin/activate

# Unittests
python3 -m unittest discover --verbose 0 --start-directory test

# Flake 8
flake8 --max-line-length=120 --show-source --ignore W605,E402,W503 autoPyTorch test examples --exclude autoPyTorch/api

# mypy
bash ci_scripts/run_mypy.sh
