from itertools import combinations

import numpy as np

from pure.result import Err, Ok, Result


def generates_portfolios_by_idxs(assets_per_portfolio: int, total_assets: int) -> Result[np.ndarray, str]:
    """
    Generates all possible combinations of portfolios given the number of assets per portfolio
    and the total number of assets.

    Args:
        assets_per_portfolio (int): Number of assets in each portfolio.
        total_assets (int): Total number of available assets.

    Returns:
        Result[list[list[int]], str]: A Result object containing a list of asset combinations
                                       on success, or an error message on failure.
    """
    if assets_per_portfolio <= 0 or total_assets <= 0:
        return Err("Both assets_per_portfolio and total_assets must be greater than zero.")

    if assets_per_portfolio > total_assets:
        return Err("assets_per_portfolio cannot be greater than total_assets.")

    portfolios = list(combinations(range(total_assets), assets_per_portfolio))

    return Ok(np.array(portfolios))
