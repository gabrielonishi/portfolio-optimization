import pathlib
import sys

from impure import config, data_loader
from pure.result import Err


def main() -> None:
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = config.setup_logging(
        root_dir / "config" / "logging.yaml", logging_name="dev")

    settings = config.get_settings(
        root_dir / "config" / "settings.yaml", env='PROD')

    start_date = settings['start_date']
    end_date = settings['end_date']
    data = data_loader._calculate_daily_returns(
        'MMM',
        start_date=start_date,
        end_date=end_date
    )
    if isinstance(data, Err):
        return logger.error(data.error)

    logger.debug(
        f"Daily returns for ticker 'MMM' from {start_date} to {end_date}: {data.value}"
    )


if __name__ == "__main__":
    main()
