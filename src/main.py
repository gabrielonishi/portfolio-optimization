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
        logger.info("Daily returns data loaded successfully.")

    elif ENV == 'TEST':
        daily_returns_dict = data_loader.load_daily_returns_from_file(
            root_dir / "data" / "daily_returns.pkl"
        )
        if isinstance(daily_returns_dict, Err):
            logger.error(daily_returns_dict.error)
            sys.exit(1)
        logger.info("Daily returns data loaded successfully from file.")

    weights = pure.simulate.generate_weights(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        max_weight_per_asset=settings['MAX_WEIGHT_PER_ASSET'],
        num_simulated_weights=settings['NUM_SIMULATED_WEIGHTS']
    )
    if isinstance(weights, Err):
        logger.error(weights.error)
        sys.exit(1)
    logger.info("Weights generated successfully.")
    logger.info(f"Generated {len(weights.value)} weight combinations.")


if __name__ == "__main__":
    main()
