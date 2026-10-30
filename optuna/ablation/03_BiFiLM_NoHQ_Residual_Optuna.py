import sys
from ablation_optuna_core import main

if __name__ == "__main__":
    main(["--variant", "no_hq", *sys.argv[1:]])
