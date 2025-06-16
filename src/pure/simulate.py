import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import time
from itertools import combinations
from multiprocessing.shared_memory import SharedMemory

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
        shm_info: np.ndarray,
        Rf_yearly: float = 0.05) -> Result[tuple[float, np.ndarray], str]:

    shm_name, shape, dtype = shm_info
    shm = SharedMemory(name=shm_name)
    daily_returns_matrix = np.ndarray(
        shape,
        dtype=dtype,
        buffer=shm.buf
    )

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

    shm.close()

    return Ok((SR[optimal_idx], weights[optimal_idx]))


def maximize_sharpe_aux(  # noqa: PLR0914
        assets_per_portfolio: int,
        max_weight_per_asset: float,
        num_simulated_weights: int,
        tickers_idxs: np.ndarray,
        shm_info: np.ndarray
) -> Result[tuple[float, np.ndarray, np.ndarray], str]:
    """
    Auxiliary function to maximize the Sharpe ratio for a given set of assets.

    Args:
        assets_per_portfolio (int): Number of assets in each portfolio.
        max_weight_per_asset (float): Maximum weight allowed for each asset.
        num_simulated_weights (int): Number of weight combinations to generate.
        tickers_idxs (np.ndarray): Indices of the assets in the portfolio.
        daily_returns_matrix (np.ndarray): Daily returns matrix.

    Returns:
        Result[tuple[float, np.ndarray], str]: A Result object containing the maximum Sharpe ratio
                                                and the corresponding weights on success, or an error
                                                message on failure.
    """
    start = time.time()
    times_1 = []
    times_2 = []

    max_sharpe = float('-inf')
    optimal_weights = np.array([])
    optimal_tickers_idxs = np.array([])

    for ticker_idx in tickers_idxs:
        weights = generate_weights(
            assets_per_portfolio, max_weight_per_asset, num_simulated_weights
        )
        if isinstance(weights, Err):
            return weights

        # sharpe_result = maximize_sharpe(
        #     ticker_idx,
        #     weights.value,
        #     shm_info
        # )
        # if isinstance(sharpe_result, Err):
        #     return sharpe_result
        # sharpe, weights = sharpe_result.value

        weights = weights.value
        Rf_yearly = 0.05

        shm_name, shape, dtype = shm_info
        shm = SharedMemory(name=shm_name)
        daily_returns_matrix = np.ndarray(
            shape,
            dtype=dtype,
            buffer=shm.buf
        )

        ANNUALIZATION_FACTOR = 252

        R_daily = daily_returns_matrix[:, ticker_idx]
        t1 = time.time()
        Rp_daily = np.dot(R_daily, weights.T)
        times_1.append(time.time() - t1)
        Rp_yearly = np.mean(Rp_daily, axis=0) * ANNUALIZATION_FACTOR
        ER_yearly = Rp_yearly - Rf_yearly
        cov_matrix = np.cov(R_daily, rowvar=False)
        t2 = time.time()
        Var_daily = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
        times_2.append(time.time() - t2)
        Vol_yearly = np.sqrt(Var_daily * ANNUALIZATION_FACTOR)

        SR = ER_yearly / Vol_yearly

        optimal_idx = np.argmax(SR)
        s, w = SR[optimal_idx], weights[optimal_idx]

        if s > max_sharpe:
            max_sharpe = s
            optimal_weights = w
            optimal_tickers_idxs = ticker_idx

        shm.close()

    print(
        f'Process {mp.current_process().name} completed at {time.strftime("%H:%M:%S")}\n'
        f'\tTotal time: {time.time() - start:.2f} seconds\n'
        f'\tAvg t1: {np.mean(times_1):.2f} seconds\n'
        f'\tAvg t2: {np.mean(times_2):.2f} seconds\n'
    )
    return Ok((max_sharpe, optimal_weights, optimal_tickers_idxs))


def run(  # noqa: PLR0913, PLR0917
        assets_per_portfolio: int,
        total_assets: int,
        max_weight_per_asset: float,
        num_simulated_weights: int,
        daily_returns_matrix: np.ndarray,
        num_simulations: int | None = None
) -> Result[tuple[float, np.ndarray, np.ndarray], str]:
    """
    Runs the simulation to maximize the Sharpe ratio.

    Args:
        assets_per_portfolio (int): Number of assets in each portfolio.
        total_assets (int): Total number of available assets.
        max_weight_per_asset (float): Maximum weight allowed for each asset.
        num_simulated_weights (int): Number of weight combinations to generate.
        daily_returns_matrix (np.ndarray): Daily returns matrix.

    Returns:
        Result[tuple[float, np.ndarray], str]: A Result object containing the maximum Sharpe ratio
                                                and the corresponding weights on success, or an error
                                                message on failure.
    """

    portfolios_idxs = generates_portfolios_by_idxs(
        assets_per_portfolio, total_assets
    )
    if isinstance(portfolios_idxs, Err):
        return portfolios_idxs

    if num_simulations is not None:
        portfolios_idxs = portfolios_idxs.value[:num_simulations]
    else:
        portfolios_idxs = portfolios_idxs.value

    n_processes = mp.cpu_count()
    # n_processes = 1

    batch_size = len(portfolios_idxs) // n_processes + 1
    idxs_batches = [
        portfolios_idxs[i:i + batch_size] for i in range(0, len(portfolios_idxs), batch_size)
    ]
    daily_returns_matrix_t = daily_returns_matrix.T

    shm = SharedMemory(create=True, size=daily_returns_matrix_t.nbytes)
    shm_returns = np.ndarray(
        daily_returns_matrix_t.shape,
        dtype=daily_returns_matrix_t.dtype,
        buffer=shm.buf
    )
    np.copyto(shm_returns, daily_returns_matrix_t)

    shm_info = (
        shm.name,
        daily_returns_matrix_t.shape,
        daily_returns_matrix_t.dtype
    )

    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(
            maximize_sharpe_aux,
            [
                (
                    assets_per_portfolio,
                    max_weight_per_asset,
                    num_simulated_weights,
                    batch,
                    shm_info
                )
                for batch in idxs_batches
            ]
        )

    shm.close()
    shm.unlink()

    print(f"Number of results: {len(results)}")

    max_sharpe = float('-inf')
    optimal_weights = np.array([])
    optimal_tickers_idxs = np.array([])

    for result in results:
        if isinstance(result, Err):
            return result

        sharpe, weights, tickers_idxs = result.value
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            optimal_weights = weights
            optimal_tickers_idxs = tickers_idxs

    return Ok((max_sharpe, optimal_weights, optimal_tickers_idxs))
