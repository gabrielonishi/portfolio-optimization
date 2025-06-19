import io
import logging
import pathlib
import pickle
from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import pure.data_loader as utils
from pure.result import Err, Ok, Result

logger = logging.getLogger(__name__)


def _fetch_tables_from_url(url: str) -> Result[list[pd.DataFrame], str]:
    """
    Fetches all tables from a given URL and returns it as a pandas DataFrame.

    Args:
        url (str): The URL to fetch the table from.

    Returns:
        Result[list[pd.DataFrame], str]: A Result object containing a list of
                                         DataFrames on success, or an error
                                         message on failure.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content.decode('utf-8')
        tables = pd.read_html(io.StringIO(content))
        if tables:
            return Ok(tables)
        else:
            return Err(f"No tables found at {url}")
    except requests.exceptions.RequestException as e:
        return Err(f"Failed to fetch data from {url}: {e}")


def _fetch_tickers_from_wikipedia() -> Result[list[str], str]:

    DOW_JONES_WIKI_URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

    tables = _fetch_tables_from_url(DOW_JONES_WIKI_URL)
    if isinstance(tables, Err):
        logger.error(tables.error)
        return tables

    tickers = utils.get_tickers_from_table(tables.value)
    if isinstance(tickers, Err):
        logger.error(tickers.error)
        return tickers

    return Ok(tickers.value)


def _load_ticker_data(ticker: str, start_date: date, end_date: date) -> Result[pd.DataFrame, str]:
    """
    Loads historical stock data for a given ticker symbol between specified dates.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (date): The start date for the data.
        end_date (date): The end date for the data.

    Returns:
        Result[pd.DataFrame, str]: A Result object containing the DataFrame with stock data on success,
                                    or an error message on failure.
    """
    start = start_date.isoformat()
    end = end_date.isoformat()
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None:
            return Err(f"Failed to download data for ticker {ticker}.")
        if df.empty:
            return Err(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
        return Ok(df)
    except Exception as e:
        return Err(f"Failed to load data for ticker {ticker}: {e}")


def _calculate_daily_returns(ticker: str, start_date: date, end_date: date):
    ticker_data = _load_ticker_data(ticker, start_date, end_date)

    if isinstance(ticker_data, Err):
        logger.error(ticker_data.error)
        return ticker_data

    ticker_df = ticker_data.value
    array_returns = utils.calculate_daily_returns(ticker, ticker_df)

    return Ok(array_returns)


def _build_daily_returns_dict(
        tickers: list[str], start_date: date, end_date: date) -> Result[dict[str, np.ndarray], str]:
    """
    Builds a dictionary of daily returns for each ticker symbol.

    Args:
        tickers (list[str]): List of stock ticker symbols.
        start_date (date): The start date for the data.
        end_date (date): The end date for the data.

    Returns:
        Result[dict[str, np.ndarray], str]: A Result object containing a dictionary of daily returns on success,
                                             or an error message on failure.
    """
    results = {}
    for ticker in tickers:
        result = _calculate_daily_returns(ticker, start_date, end_date)
        if isinstance(result, Err):
            logger.error(result.error)
            results[ticker] = result
        else:
            results[ticker] = result.value

    return Ok(results)


def _save_daily_returns_to_file(
        daily_returns: dict[str, np.ndarray], file_path: pathlib.Path) -> Result[None, str]:
    """
    Saves the daily returns dictionary to a file using pickle.

    Args:
        daily_returns (dict[str, np.ndarray]): Dictionary of daily returns.
        file_path (str): Path to the file where the data will be saved.

    Returns:
        Result[None, str]: A Result object indicating success or failure.
    """
    try:
        with open(file_path, mode='wb') as f:
            pickle.dump(daily_returns, f)
        return Ok(None)
    except Exception as e:
        return Err(f"Failed to save daily returns to {file_path}: {e}")


def load_daily_returns_from_file(file_path: pathlib.Path) -> Result[tuple[list[str], np.ndarray], str]:
    """
    Loads daily returns from a file using pickle.

    Args:
        file_path (str): Path to the file where the data is saved.

    Returns:
        Result[dict[str, np.ndarray], str]: A Result object containing the daily returns dictionary on success,
                                             or an error message on failure.
    """
    try:
        with open(file_path, mode='rb') as f:
            daily_returns_dict = pickle.load(f)
            daily_returns_matrix = np.stack([daily_returns_dict[ticker]
                                             for ticker in daily_returns_dict.keys()])
            tickers = list(daily_returns_dict.keys())
        return Ok((tickers, daily_returns_matrix))
    except Exception as e:
        return Err(f"Failed to load daily returns from {file_path}: {e}")


def run(start_date: date,
        end_date: date,
        saved_copy_filepath: pathlib.Path | None = None) -> Result[tuple[list[str], np.ndarray], str]:

    tickers = _fetch_tickers_from_wikipedia()
    if isinstance(tickers, Err):
        return tickers

    daily_returns_dict = _build_daily_returns_dict(
        tickers.value, start_date, end_date)
    if isinstance(daily_returns_dict, Err):
        return daily_returns_dict

    print(daily_returns_dict)

    if saved_copy_filepath:
        save_result = _save_daily_returns_to_file(
            daily_returns_dict.value, saved_copy_filepath)
        if isinstance(save_result, Err):
            return save_result

    daily_returns_matrix = np.stack([daily_returns_dict.value[ticker]
                                     for ticker in daily_returns_dict.value.keys()])

    tickers = list(daily_returns_dict.value.keys())

    return Ok((tickers, daily_returns_matrix))
