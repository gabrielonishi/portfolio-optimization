import logging
import pathlib
import sys

from impure import logging_config


def main() -> None:
    root_dir = pathlib.Path(sys.argv[0]).parent.parent.resolve()
    logger = logging_config.setup_logging(root_dir / "config" / "logging.yaml", logging_name="dev")
    logger.info("Logging setup complete.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")


if __name__ == "__main__":
    main()
