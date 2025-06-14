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

    combinations = pure.simulate.generates_portfolios_by_idxs(
        assets_per_portfolio=settings['ASSETS_PER_PORTFOLIO'],
        total_assets=settings['TOTAL_ASSETS']
    )
    if isinstance(combinations, Err):
        logger.error(combinations.error)
        sys.exit(1)

    logger.debug(combinations.value.shape)


if __name__ == "__main__":
    main()
