"""Rank Risk-Reward Settings by HIGHEST WIN RATE across 1m, 2m, 3m, 5m timeframes."""

import pandas as pd
from grid_search_fast_pointer import main as run_pointer_grid_search

if __name__ == "__main__":
    # Import and run fast pointer search, ranking by WIN RATE
    import grid_search_fast_pointer
    grid_search_fast_pointer.main()
