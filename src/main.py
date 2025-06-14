import pathlib
import sys

from impure import config, data_loader
from pure.result import Err

ENV = 'TEST'


def main() -> None:
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = config.setup_logging(
        root_dir / "config" / "logging.yaml", logging_name="dev")

    if ENV == 'PROD':
        settings = config.get_settings(
            root_dir / "config" / "settings.yaml", env='PROD')

        start_date = settings['start_date']
        end_date = settings['end_date']

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

    print(daily_returns_dict.value)


if __name__ == "__main__":
    main()
