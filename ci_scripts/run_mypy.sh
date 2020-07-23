#MYPYPATH=autoPyTorch
MYPYOPTS=""

MYPYOPS="$MYPYOPS --ignore-missing-imports --follow-imports skip"
MYPYOPTS="$MYPYOPS --disallow-untyped-decorators"
MYPYOPTS="$MYPYOPS --disallow-incomplete-defs"
MYPYOPTS="$MYPYOPS --disallow-untyped-defs"

mypy $MYPYOPTS --show-error-codes autoPyTorch
