"""Export a local Adanos Reddit crypto token response for offline analysis."""

import csv
from datetime import date
import json
from pathlib import Path
import re

import click


OPTIONAL_FIELDS = {
    "buzz_score": ((int, float), 0, 100),
    "sentiment_score": ((int, float), -1, 1),
    "bullish_pct": ((int,), 0, 100),
    "bearish_pct": ((int,), 0, 100),
}
FIELDS = ["timestamp", "mentions", *OPTIONAL_FIELDS]


def daily_trend_rows(payload, symbol):
    if not re.fullmatch(r"\$?[A-Za-z0-9]{1,20}", symbol):
        raise ValueError("Use an explicit Adanos token symbol, such as BTC.")
    expected = symbol.lstrip("$").upper()
    if not isinstance(payload, dict):
        raise ValueError("Expected a Reddit crypto token response object.")
    actual = payload.get("symbol")
    if not isinstance(actual, str) or actual.upper() != expected:
        raise ValueError("Response symbol does not match the requested token.")
    if payload.get("found") is False:
        raise ValueError("No qualifying data for this token and window (found=false).")
    if payload.get("found") is not True:
        raise ValueError("Expected the response found flag to be boolean true.")
    trend = payload.get("daily_trend")
    if not isinstance(trend, list) or not trend:
        raise ValueError("No daily history to export.")

    rows = []
    dates = set()
    for item in trend:
        if not isinstance(item, dict):
            raise ValueError("Each daily history item must be an object.")
        value = item.get("date")
        try:
            day = date.fromisoformat(value)
        except (TypeError, ValueError):
            raise ValueError("Each date must be a valid YYYY-MM-DD UTC date.") from None
        if day.isoformat() != value:
            raise ValueError("Each date must use YYYY-MM-DD format.")
        if day in dates:
            raise ValueError("Daily history contains duplicate dates.")
        dates.add(day)
        mentions = item.get("mentions")
        if type(mentions) is not int or mentions < 0:
            raise ValueError("Daily mentions must be a nonnegative integer.")
        row = {"timestamp": day.isoformat() + "T00:00:00", "mentions": mentions}
        for field, (types, lower, upper) in OPTIONAL_FIELDS.items():
            value = item.get(field)
            if value is not None and (type(value) not in types or not lower <= value <= upper):
                raise ValueError("Invalid daily " + field + " type or range.")
            row[field] = value
        rows.append(row)
    return sorted(rows, key=lambda row: row["timestamp"])


@click.command()
@click.option("--input-file", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Local JSON response from /reddit/crypto/v1/token/{symbol}.")
@click.option("--output-file", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="New CSV path. The parent directory must already exist.")
@click.option("--symbol", required=True, help="Explicit Adanos token symbol, such as BTC.")
def main(input_file, output_file, symbol):
    try:
        with input_file.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, ValueError):
        raise click.ClickException("Cannot read input as UTF-8 JSON.") from None
    try:
        rows = daily_trend_rows(payload, symbol)
    except ValueError as error:
        raise click.ClickException(str(error)) from None
    try:
        with output_file.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError:
        raise click.ClickException("Output already exists. Choose a new destination.") from None
    except OSError:
        raise click.ClickException("Cannot create CSV output. Check the parent directory and permissions.") from None
    click.echo(f"Exported {len(rows)} daily rows for offline analysis.")


if __name__ == "__main__":
    main()
