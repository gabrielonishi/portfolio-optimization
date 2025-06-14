import pathlib
import sys

import numpy as np

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

    portfolios_idxs = pure.simulate.generates_portfolios_by_idxs(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        total_assets=settings['TOTAL_ASSETS']
    )
    if isinstance(portfolios_idxs, Err):
        logger.error(portfolios_idxs.error)
        sys.exit(1)

    portfolios_idxs = portfolios_idxs.value

    if ENV == 'PROD':

        start_date = settings['PROD']['start_date']
        end_date = settings['PROD']['end_date']

        daily_returns_dict = data_loader.run(
            start_date=start_date,
            end_date=end_date,
            saved_copy_filepath=root_dir / "data" / "daily_returns.pkl"
        )

        if isinstance(daily_returns_dict, Err):
            logger.error(daily_returns_dict.error)
            sys.exit(1)

    elif ENV == 'TEST':
        num_simulations = settings['TEST']['NUM_SIMULATIONS']
        portfolios_idxs = portfolios_idxs[:num_simulations]
        daily_returns_dict = data_loader.load_daily_returns_from_file(
            root_dir / "data" / "daily_returns.pkl"
        )

        if isinstance(daily_returns_dict, Err):
            logger.error(daily_returns_dict.error)
            sys.exit(1)

    weights = pure.simulate.generate_weights(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        max_weight_per_asset=settings['MAX_WEIGHT_PER_ASSET'],
        num_simulated_weights=settings['NUM_SIMULATED_WEIGHTS']
    )

    if isinstance(weights, Err):
        logger.error(weights.error)
        sys.exit(1)

    daily_returns_matrix = np.stack([daily_returns_dict.value[ticker]
                                     for ticker in daily_returns_dict.value.keys()])

    sharpe = pure.simulate.maximize_sharpe(
        tickers_idxs=portfolios_idxs[0],
        weights=weights.value,
        daily_returns_matrix=daily_returns_matrix
    )

    if isinstance(sharpe, Err):
        logger.error(sharpe.error)
        sys.exit(1)

    print(
        f"Sharpe ratio for the first portfolio: {sharpe.value[0]:.4f}"
        f" with weights: {sharpe.value[1]}"
    )


if __name__ == "__main__":
    main()
