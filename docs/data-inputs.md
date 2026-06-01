# Data sources and data collectors

## Defining data sources

The intelligent trading bot operates in two modes:
-   **Batch (offline) mode:** Intended for analyzing large historical datasets.
-   **Stream (online) mode:** Intended for generating regular predictions on small increments of new data.

In batch mode, historical data must be retrieved for analysis. In stream mode, only the latest data is retrieved and analyzed incrementally. In both cases, the data structure must remain consistent (except in training mode, where labels must also be generated).

Data sources for both modes are specified in the `data_sources` section. Each entry in this section describes a single data source used to retrieve data.

```jsonc
"data_sources": [
  {...}, // First data source
  {...}, // Second data source
  {...} // Third data source
]
```

A data source description includes the following attributes:

```jsonc
{
  "folder": "ETHUSDT", // Quote name as defined by the data provider; also serves as the folder name
  "file": "klines", // Filename for the source data
  "column_prefix": "etn" // Prefix added to all columns from this data source
}
```

The attributes of a data source are interpreted as follows:

-   `folder`: This attribute serves two purposes: it specifies the folder where data is stored locally and the quote name (symbol) used to request data from the provider.
-   `file`: The name of the file containing the retrieved data. For example, candlestick data might use `klines`. If not specified, it defaults to the symbol name defined in the `folder` attribute.
-   `column_prefix`: When retrieving multiple symbols, column names often overlap (e.g., open, high, low, close). To distinguish the origin of these columns after merging them into a single DataFrame, use the `column_prefix` attribute. This prefix is applied to every column name from this source during the merge process. Note that the prefix is used only for merging; the original column names in the source file remain unchanged.

Below is an example configuration with two data sources:

```jsonc
"data_sources": [
  {"folder": "ETHUSDT", "file": "klines", "column_prefix": ""},
  {"folder": "ETHBTC", "file": "klines", "column_prefix": "ethbtc"}
]
```

In this example, the first data source retrieves quotes for ETH. The source data is stored in a file named `klines` (the file extension depends on the chosen format). Because no prefix is specified, the columns retain their original names when merged into the main DataFrame. The second data source provides the Ethereum-to-Bitcoin price, which serves as additional analytical data. A column prefix is required here to distinguish its columns from those of the first data source.

When retrieving data, you must specify the frequency (time raster). This frequency applies to all data sources and is defined in the `freq` attribute of the configuration file. Values for this attribute follow the `pandas` offset alias convention: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

For example, `h` represents hourly frequency, `min` represents minutely frequency, and `D` represents calendar day frequency. A number preceding the alias indicates the duration of one period (e.g., `15min` means every 15 minutes). This frequency string is subsequently converted to the representation expected by the specific data provider, if supported.

Credentials for accessing the data provider are loaded by the provider-specific component from the configuration file. For Binance, for instance, the `api_key` and `api_secret` attributes are used. Custom arguments for the client are specified in the configuration as a dictionary:

```jsonc
"client_args": {"tld": "us"} // Connects to the country-specific Binance API server
```

The data provider is specified in the `venue` attribute. Currently, the following values are supported:

-   `binance`: Binance
-   `mt5`: MT5
-   `yahoo`: Yahoo

## Downloader

The `download` script retrieves data from the configured data sources and stores it in the corresponding files. Currently, CSV format is used. If the target file already exists, only the latest data is retrieved and appended; existing rows are overwritten in case of overlap. If the file does not exist, the maximum available history is retrieved. The maximum stored size is controlled by the `download_max_rows` attribute.

Execute the downloader script as follows:

```console
python -m scripts.download -c config.json
```

If the configuration file defines two data sources with the required attributes, the script will download two files and store them in their respective folders.

## Merging data sources

Downloaded data from different sources is not used in isolation. Instead, it is merged into a single table via a merge procedure implemented in both the merge script and the server. The merge procedure has two primary goals:

-   Generate a continuous time raster based on the configured frequency to prevent gaps in the source data.
-   Append all source data (columns) to this table by aligning rows with the generated raster.

Execute the merge script as follows:

```console
python -m scripts.merge -c config.json
```

The result is saved as a single file containing data from all sources. The output filename (and format) is specified in the `merge_file_name` attribute of the configuration file. For example, to store the merged data in Parquet format, use: `"merge_file_name": "data.parquet"`.

In online mode, the server merges data for each new request (e.g., every minute) after retrieving chunks from all data sources. This merged data is then appended to the analyzer's main DataFrame. Columns from the merged table can be referenced in [feature definitions](features.md).

## Implementing a custom data collector

To implement a new custom data collector for a specific data provider, perform the following steps:

-   Add a new entry to the `Venue` enumerator.
-   Implement the provider-specific functions responsible for data retrieval: `fetch_klines`, `health_check`, and `download_klines`.
-   Return these functions from the dispatcher functions `get_collector_functions` and `get_download_functions`.

The server dynamically locates these functions based on the venue specified in the configuration. It uses them to incrementally retrieve data, merge it, append it to the main DataFrame, and perform analysis.
