import pandas as pd
from typing import List, Dict, Any


def merge_data_sources(data_sources: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Merges multiple data sources into a single DataFrame based on the configured time column and frequency.

    Args:
        data_sources (List[Dict]): A list of dicts where each dict includes a DataFrame under the "df" key,
                                   along with metadata such as "column_prefix".
        config (Dict): Configuration dictionary containing:
            - "time_column" (str): Name of the timestamp column
            - "freq" (str): Pandas frequency string (e.g. "1h")
            - "merge_interpolate" (bool, optional): Whether to interpolate numeric columns

    Returns:
        pd.DataFrame: A merged DataFrame with a regular time index.
    """
    time_column = config["time_column"]
    freq = config["freq"]

    # Process each data source
    for source in data_sources:
        df = source["df"]

        # Ensure time column is the index
        if time_column in df.columns:
            df = df.set_index(time_column)
        elif df.index.name != time_column:
            raise ValueError(f"Missing or misaligned time index in source: expected '{time_column}'")

        # Apply column prefix if defined
        prefix = source.get("column_prefix", "")
        if prefix:
            df.columns = [
                f"{prefix}_{col}" if not col.startswith(f"{prefix}_") else col
                for col in df.columns
            ]

        # Store processed df and its time range
        source["df"] = df
        source["start"] = df.first_valid_index()
        source["end"] = df.last_valid_index()

    # Determine common time range for all sources
    common_start = min(src["start"] for src in data_sources)
    common_end = min(src["end"] for src in data_sources)

    # Create unified time index
    unified_index = pd.date_range(start=common_start, end=common_end, freq=freq)
    merged_df = pd.DataFrame(index=unified_index)
    merged_df.index.name = time_column

    # Join all data sources
    for source in data_sources:
        merged_df = merged_df.join(source["df"], how="left")

    # Interpolate numeric columns if enabled
    if config.get("merge_interpolate", False):
        numeric_cols = merged_df.select_dtypes(include=["float", "int"]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].interpolate()

    return merged_df
