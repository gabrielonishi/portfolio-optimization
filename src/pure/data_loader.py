import logging

import numpy as np
import pandas as pd

from pure.result import Err, Ok, Result

logger = logging.getLogger(__name__)


def get_tickers_from_table(tables: list[pd.DataFrame]) -> Result[list[str], str]:
    """
    Extracts tickers from a given DataFrame.

    Args:
        table (pd.DataFrame): The DataFrame containing the tickers.

    Returns:
        Result[list[str], str]: A Result object containing the list of tickers on success,
                                or an error message on failure.
    """

    for table in tables:
        if 'Company' in table and 'Exchange' in table and 'Symbol' in table:
            return Ok(list(table['Symbol']))

    return Err("No valid table found with 'Company', 'Exchange', and 'Symbol' columns.")


def calculate_daily_returns(ticker: str, ticker_df: pd.DataFrame) -> Result[np.ndarray, str]:
    if 'Close' not in ticker_df.columns or ticker_df['Close'].empty:
        return Err(f"No 'Close' column found in data for ticker {ticker}.")

    daily_returns = ticker_df['Close'] / ticker_df['Close'].shift(1) - 1
    array_returns = np.array(daily_returns[ticker][1:], dtype=np.float64)

    return Ok(array_returns)
