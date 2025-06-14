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


def generate_weights(
        assets_per_portfolio: int,
        max_weight_per_asset: float,
        num_simulated_weights: int) -> Result[np.ndarray, str]:
    '''
    Generates a matrix of random weights for portfolios.
    Args:
        assets_per_portfolio (int): Number of assets in each portfolio.
        max_weight_per_asset (float): Maximum weight allowed for each asset.
        num_simulated_weights (int): Number of weight combinations to generate.
    Returns:
        Result[np.ndarray, str]: A Result object containing a matrix of weights on success,
                                  or an error message on failure.
    '''

    if 1 / assets_per_portfolio > max_weight_per_asset:
        return Err(
            f"Max weight {max_weight_per_asset} is too low for {assets_per_portfolio} assets. "
            f"Minimum weight per asset must be at least {1 / assets_per_portfolio}."
        )

    weights_matrix = np.empty(
        (num_simulated_weights, assets_per_portfolio), dtype=np.float64)
    portfolio_size = np.ones(assets_per_portfolio)

    BATCH_SIZE_MULTIPLIER = 1.2

    i = 0

    while i < num_simulated_weights:
        spots_left = num_simulated_weights - i
        batch_size = int(BATCH_SIZE_MULTIPLIER * spots_left)
        batch = np.random.dirichlet(alpha=portfolio_size, size=batch_size)
        valid_weights = batch[np.all(batch <= max_weight_per_asset, axis=1)]
        spots_to_be_filled = min(len(valid_weights), num_simulated_weights - i)

        if spots_to_be_filled > 0:
            weights_matrix[i:i +
                           spots_to_be_filled] = valid_weights[:spots_to_be_filled]
            i += spots_to_be_filled

    return Ok(weights_matrix)


def maximize_sharpe(
        tickers_idxs: np.ndarray,
        weights: np.ndarray,
        daily_returns_matrix: np.ndarray,
        Rf_yearly: float = 0.05) -> Result[tuple[float, np.ndarray], str]:

    ANNUALIZATION_FACTOR = 252

    R_daily = daily_returns_matrix[tickers_idxs].T
    Rp_daily = R_daily @ weights.T
    Rp_yearly = np.mean(Rp_daily, axis=0) * ANNUALIZATION_FACTOR
    ER_yearly = Rp_yearly - Rf_yearly

    cov_matrix = np.cov(R_daily, rowvar=False)
    Var_daily = np.diag(weights @ cov_matrix @ weights.T)
    Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

    SR = ER_yearly / Vol_yearly

    optimal_idx = np.argmax(SR)

    return Ok((SR[optimal_idx], weights[optimal_idx]))
