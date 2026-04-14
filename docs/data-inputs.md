# Data sources and data collectors

## Defining data sources

The intelligent trading bot works in two modes:
- batch or offline mode intended for analyzing large historical data
- stream or online mode intended for regular predictions applied to small new data

In batch mode, the historical data has to be retrieved for analysis.
In stream mode, only the latest data has to be retrieved and incrementally analyzed.
In both cases, the structure of data must be the same (except that in train mode, label have to be additionally generated).

The data sources for the both modes are specified in the `data_sources` section.
Each entry of this section describes one data source which will be used to retrieve data.
```jsonc
"data_sources": [
  {...}, // First data source
  {...}, // Second data source
  {...} // Third data source
]
```

One data source description has the following attributes:
```jsonc
{
  "folder": "ETHUSDT", // Quote name as defined by data provider and folder name
  "file": "klines", // File name for the source data
  "column_prefix": "etn" // Added to all columns from this data source
 }
```

The attributes of a data source have the following interpretations:
- `folder`: It has two uses: folder name where the data is located and quote name used to request the data.
In other words, it is equal to symbol name as defined by the data provider. Simultaneously, it is where the retrieved data is located.
- `file`: It is name of the file with the retrieved data. For example, for candle stick data, it can be `klines`.
If not specified, it is equal to the symbol name in the `folder` attribute.
- `column_prefix`: If we retrieve different symbols then they may have the same column names, typically, open, high, low, close.
To distinguish the origin of these columns after merging into one common dataframe, the attribute `column_prefix` is used.
It will added to every column name from this data source.
Note that this prefix will be used only for merging while data in the source file will have the original column names.

Here is an example of two data sources:
```jsonc
"data_sources": [
  {"folder": "ETHUSDT", "file": "klines", "column_prefix": ""},
  {"folder": "ETHBTC", "file": "klines", "column_prefix": "ethbtc"}
]
```
Here the first data source is used to retrieve the quotes for ETH.
The source data will be stored in the file `klines` (file extensions will be chosen depending on the file format).
No prefix is specified and hence the columns will have their original name when merged into one dataframe.
The second data source describes Ethereum to Bitcoin price which we want to use as additional data for analysis.
Here it is necessary to specify column prefix in order to distinguish its columns from those of the first data source.

We retrieving data it is necessary to know frequency (time raster).
It is the same for all data sources and is specified in the `freq` attribute of the configuration file.
The values of this attribute follow `pandas` convention described here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
For example, `h` is hourly frequency, `min` is minutely frequency and `D` is calendar day frequency.
The number before the alias is how many hours, minutes, days etc. is included in one period.
For example, `15min` means every 15 minutes.
This frequency string will be then converted to the representation expected by one or another data provider (if supported).

Credentials to access the data provider are loaded by the provider specific component from the configuration file.
For example, for Binance, these attributes are used: `api_key` and `api_secret`.
Custom arguments for the client is specified in section of the configuration as a dictionary:
```jsonc
  "client_args", {"tld": "us"} // To country-specific Binance API server
```

Data provider is specified in the `venue` attribute. Currently these values are supported:
- `binance` Binance
- `mt5` MT5
- `yahoo` Yahoo

## Downloader

The `download` script is intended for downloading data from the data sources and storing them in the corresponding files.
Currently CSV format is used. If the file already exists then only the latest data will be retrieved and appended
to the file by overwriting existing rows in case of overlap.
If the file does not exist, then maximum length will be retrieved.
The maximum stored size is specified in the `download_max_rows` attribute.

The downloader is executed as a script as follows:
```console
python -m scripts.download -c config.json
```

If the configuration file has two data sources and the required attributes then it will download two files
and store them in the specified folders.

## Merging data sources

The downloaded data from different data sources are not used separately.
Instead, they are merged into one table by the merge procedure implemented as a script and in the server.
The merge procedure has two major goals:
- Generate continuous time raster according to the frequency in order to avoid gaps in the source data
- Append all source data (columns) to this table by aligning their rows with this raster

The result of merge is stored as a file by the script or maintained as a datafrme by the analyzer.




The result is a dataframe with all source columns as retrieved from the data sources and named accordingly.
These columns can be then referenced from feature definitions (see feature definitions).

The merge script is executed as follows:
```console
python -m scripts.merge -c config.json
```

The result is stored as one file with the data from all source files.
The output file name (and format) is specified in the `merge_file_name` attribute of the configuration file.
For example, if we want to store the merged data in `parquet` format then we use: `"merge_file_name": "data.parquet"`

The server in online mode will merge data for each new request (for example, every minute)
after retrieving chunks from all the data sources, and then append this merged data to the main dataframe of the analyzer.

## Implementing a custom data collector

TBD
