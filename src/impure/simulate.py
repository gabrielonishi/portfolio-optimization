import multiprocessing as mp

import numpy as np

import pure.simulate as utils
from pure.result import Err, Ok, Result
from pure.safe_shared_memory import SafeSharedMemory


def maximize_sharpe(
        tickers_idxs: np.ndarray,
        weights: np.ndarray,
        shm_info: np.ndarray,
        Rf_yearly: float = 0.05) -> Result[tuple[float, np.ndarray], str]:

    shm_name, shape, dtype = shm_info

    with SafeSharedMemory(name=shm_name, unlink=False) as shm:
        daily_returns_matrix = np.ndarray(
            shape,
            dtype=dtype,
            buffer=shm.buf
        )

        results = utils.maximize_sharpe(
            tickers_idxs,
            weights,
            daily_returns_matrix,
            Rf_yearly
        )

    return results if isinstance(results, Err) else Ok(results.value)


def maximize_sharpe_aux(
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

    max_sharpe = float('-inf')
    optimal_weights = np.array([])
    optimal_tickers_idxs = np.array([])

    for ticker_idx in tickers_idxs:
        rng = np.random.default_rng()
        weights = utils.generate_weights(
            assets_per_portfolio, max_weight_per_asset, num_simulated_weights, rng
        )
        if isinstance(weights, Err):
            return weights
        sharpe_result = maximize_sharpe(
            ticker_idx,
            weights.value,
            shm_info
        )
        if isinstance(sharpe_result, Err):
            return sharpe_result
        sharpe, weights = sharpe_result.value
        if sharpe > max_sharpe:
            max_sharpe = sharpe
            optimal_weights = weights
            optimal_tickers_idxs = ticker_idx

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

    portfolios_idxs = utils.generates_portfolios_by_idxs(
        assets_per_portfolio, total_assets
    )

    if isinstance(portfolios_idxs, Err):
        return portfolios_idxs

    if num_simulations is not None and num_simulations != 0:
        portfolios_idxs = portfolios_idxs.value[:num_simulations]
    else:
        portfolios_idxs = portfolios_idxs.value

    n_processes = mp.cpu_count()

    idxs_batches = []
    batch_size = len(portfolios_idxs) // n_processes

    for i in range(n_processes):
        if i == n_processes - 1:
            idxs_batches.append(portfolios_idxs[i * batch_size:])
        else:
            idxs_batches.append(portfolios_idxs[i * batch_size:(i + 1) * batch_size])

    with SafeSharedMemory(
            create=True,
            size=daily_returns_matrix.nbytes,
            unlink=True
    ) as shm:
        shm_returns = np.ndarray(
            daily_returns_matrix.shape,
            dtype=daily_returns_matrix.dtype,
            buffer=shm.buf
        )
        np.copyto(shm_returns, daily_returns_matrix)

        shm_info = (
            shm.name,
            daily_returns_matrix.shape,
            daily_returns_matrix.dtype
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

    max_results = utils.compare_process_results(results)

    return max_results if isinstance(max_results, Err) else Ok(max_results.value)
