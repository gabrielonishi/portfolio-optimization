import io
import logging

import pandas as pd
import requests

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


def _get_tickers_from_table(tables: list[pd.DataFrame]) -> Result[list[str], str]:
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


def fetch_tickers_from_wikipedia() -> Result[list[str], str]:

    DOW_JONES_WIKI_URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'

    tables = _fetch_tables_from_url(DOW_JONES_WIKI_URL)
    if isinstance(tables, Err):
        logger.error(tables.error)
        return tables

    tickers = _get_tickers_from_table(tables.value)
    if isinstance(tickers, Err):
        logger.error(tickers.error)
        return tickers

    return Ok(tickers.value)
