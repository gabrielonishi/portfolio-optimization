import pathlib
import sys

from impure import data_loader, logging_config
from pure.result import Err


def main() -> None:
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = logging_config.setup_logging(
        root_dir / "config" / "logging.yaml", logging_name="dev")

    dow_tickers = data_loader.fetch_tickers_from_wikipedia()
    if isinstance(dow_tickers, Err):
        logger.error(dow_tickers.error)
        sys.exit(1)

    logger.debug(f"Dow Jones tickers: {dow_tickers.value}")


if __name__ == "__main__":
    main()
