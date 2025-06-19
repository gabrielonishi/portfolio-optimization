import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pathlib
import sys
import time

from impure import config, data_loader, simulate
from pure.result import Err

ENV = 'TEST'


def main() -> None:  # noqa: PLR0914
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = config.setup_logging(
        root_dir / "config" / "logging.yaml", logging_name="dev")
    settings = config.get_settings(
        root_dir / "config" / "settings.yaml")

    if ENV == 'PROD':
        start = time.time()
        num_simulations = None
        start_date = settings['PROD']['start_date']
        end_date = settings['PROD']['end_date']

        data_loader_return = data_loader.run(
            start_date=start_date,
            end_date=end_date,
            saved_copy_filepath=root_dir / "data" / "daily_returns.pkl"
        )
        logger.info(f'Fetched data in {time.time() - start:.2f} seconds')

    elif ENV == 'TEST':
        start = time.time()
        num_simulations = settings['TEST']['NUM_SIMULATIONS']
        data_loader_return = data_loader.load_daily_returns_from_file(
            root_dir / "data" / "daily_returns.pkl"
        )
        logger.info(f'Loaded data in {time.time() - start:.2f} seconds')

    if isinstance(data_loader_return, Err):
        logger.error(data_loader_return.error)
        sys.exit(1)

    num_processes = settings['NUM_PROCESSES']
    dow_tickers, daily_returns_matrix = data_loader_return.value

    start = time.time()
    logger.info("Starting simulation...")
    sharpe_result = simulate.run(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        total_assets=settings['TOTAL_ASSETS'],
        max_weight_per_asset=settings['MAX_WEIGHT_PER_ASSET'],
        num_simulated_weights=settings['NUM_SIMULATED_WEIGHTS'],
        daily_returns_matrix=daily_returns_matrix,
        num_simulations=num_simulations,
        num_processes=num_processes
    )

    logger.info("Simulation finished")

    if isinstance(sharpe_result, Err):
        logger.error(sharpe_result.error)
        sys.exit(1)

    max_sharpe, optimal_weights, optimal_tickers_idxs = sharpe_result.value
    logger.info(f"Max Sharpe Ratio: {max_sharpe}")

    portfolio_output = ''

    for ticker_idx, weight in zip(optimal_tickers_idxs, optimal_weights):
        portfolio_output += f"{dow_tickers[ticker_idx]}: {weight:.2%}\n"

    logger.info("Optimal Portfolio Weights:\n" + portfolio_output)
    logger.info(f'Simulation completed in {time.time() - start:.2f} seconds')


if __name__ == "__main__":
    main()
