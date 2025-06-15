import pathlib
import sys

import pure.simulate
from impure import config, data_loader
from pure.result import Err

ENV = 'TEST'


def main() -> None:
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = config.setup_logging(
        root_dir / "config" / "logging.yaml", logging_name="dev")
    settings = config.get_settings(
        root_dir / "config" / "settings.yaml")

    if ENV == 'PROD':
        num_simulations = None
        start_date = settings['PROD']['start_date']
        end_date = settings['PROD']['end_date']

        daily_returns_matrix = data_loader.run(
            start_date=start_date,
            end_date=end_date,
            saved_copy_filepath=root_dir / "data" / "daily_returns.pkl"
        )

    elif ENV == 'TEST':
        num_simulations = settings['TEST']['NUM_SIMULATIONS']
        daily_returns_matrix = data_loader.load_daily_returns_from_file(
            root_dir / "data" / "daily_returns.pkl"
        )

    if isinstance(daily_returns_matrix, Err):
        logger.error(daily_returns_matrix.error)
        sys.exit(1)

    sharpe_result = pure.simulate.run(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        total_assets=settings['TOTAL_ASSETS'],
        max_weight_per_asset=settings['MAX_WEIGHT_PER_ASSET'],
        num_simulated_weights=settings['NUM_SIMULATED_WEIGHTS'],
        daily_returns_matrix=daily_returns_matrix.value,
        num_simulations=num_simulations
    )
    if isinstance(sharpe_result, Err):
        logger.error(sharpe_result.error)
        sys.exit(1)
    max_sharpe, optimal_weights = sharpe_result.value
    logger.info(f"Max Sharpe Ratio: {max_sharpe}")
    logger.info(f"Optimal Weights: {optimal_weights}")


if __name__ == "__main__":
    main()
