import json
import click
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pandas.api.types import is_float_dtype
from intelligent_trading_bot.config import handle_config
from intelligent_trading_bot.common.merge import merge_data_sources
from intelligent_trading_bot.asyncbus.registry import ChannelRegistry
from intelligent_trading_bot.asyncbus import AsyncMappedBus


@click.group()
def sage_ctl():
    """
    📊 Sage CLI — Intelligent Trading Automation

    A unified control interface for managing data, ML models/pipelines, and automated trading operations.

    🛠️  Core Commands: \n
    ──────────────────────────────────────────────\n
    ⚙️  sage config     → Load and inspect configuration files \n
    🔗  sage merge      → Merge multiple data sources \n
    🧠  sage generate   → Generate features, labels and signals \n
    🏋️  sage train      → Train ML models \n
    🤖  sage predict    → Run predictions (including rolling forecasts) \n
    🧪  sage simulate   → Simulate/backtest strategies \n
    💸  sage live       → Execute trades in live/paper mode \n
    """
    pass


@sage_ctl.command(name="config", help="⚙️  Load and display the contents of a configuration file")
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help='Path to the configuration file (YAML/TOML/JSON)'
)
def read_config(config_file):
    """
    Load and display the contents of a configuration file.
    
    Supports: JSON, YAML, TOML, INI, and NumPy files.

    Example:
        sage config -c configs/example.json
    """
    try:
        config = handle_config(config_file)
    except Exception as e:
        click.secho(f"❌ Failed to load config: {e}", fg='red')
        raise SystemExit(1)

    click.secho(f"\n✅ Loaded configuration from: {config_file}\n", fg='green')
    click.echo(json.dumps(config, indent=4, default=str))



"""
Create one output file from multiple input data files. 
"""
@sage_ctl.command(name="merge")
@click.option(
    "--config-file", "-c",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    required=True,
    help="Path to the configuration file"
)
def merge(config_file):
    """
    🧩 Merge multiple data sources into a unified dataframe.

    Reads CSV data sources defined in the config, aligns them on a regular time index, 
    interpolates if specified, and stores a merged output file.
    """
    from intelligent_trading_bot.service.App import App
    
    # load_config(config_file)
    config = handle_config(config_file)
    App.config.update(config)
    time_column = App.config["time_column"]
    data_sources = App.config.get("data_sources", [])

    if not data_sources:
        click.secho("❌ ERROR: No data sources defined.", fg="red")
        return

    now = datetime.now()
    data_path = Path(App.config["data_folder"])

    for ds in data_sources:
        folder = ds.get("folder")
        file = ds.get("file", folder)
        file_path = (data_path / folder / file).with_suffix(".csv")

        if not file_path.exists():
            click.secho(f"❌ File not found: {file_path}", fg="red")
            return

        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
        click.secho(f"📄 Loaded {file_path} with {len(df)} rows", fg="green")
        ds["df"] = df

    df_merged = merge_data_sources(data_sources, App.config)

    out_path = data_path / App.config["symbol"] / App.config["merge_file_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_merged = df_merged.reset_index()
    if out_path.suffix == ".parquet":
        df_merged.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df_merged.to_csv(out_path, index=False)
    else:
        click.secho(f"❌ Unsupported output format: {out_path.suffix}", fg="red")
        return

    click.secho(f"✅ Merged file saved: {out_path} with {len(df_merged)} rows", fg="cyan")
    click.secho(f"⏱️ Completed in {str(datetime.now() - now).split('.')[0]}", fg="blue")


"""

    Generate features and labels.

"""

@click.group(name="generate", help="""
🧬 Generate derived features and supervised labels from market data, also generates signals from model predictions.

Examples: \n
  sage generate features -c config.yaml \n
  sage generate labels -c config.yaml \n
  sage generate signals -c config.json \n
""")
def generate():
    pass


@generate.command(name="features", help="⚙️ Generate engineered features from merged price data")
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, readable=True),
    required=True,
    help='Path to the configuration file'
)
@click.option('--in-nrows', type=int, default=50_000_000, show_default=True, help="Limit number of rows to load")
@click.option('--tail-rows', type=int, default=int(10.0 * 525_600), show_default=True, help="Limit number of rows to process")
def generate_features(config_file, in_nrows, tail_rows):
    """
    This command processes merged market data and applies feature engineering logic defined in the config.
    """
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.generators import generate_feature_set
    
    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    time_column = App.config["time_column"]
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("merge_file_name")

    if not file_path.is_file():
        click.secho(f"❌ Data file does not exist: {file_path}", fg="red")
        return

    click.secho(f"📥 Loading source data from: {file_path}", fg="cyan")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=in_nrows)
    else:
        click.secho(f"❌ Unsupported file format: {file_path.suffix}", fg="red")
        return

    df = df.iloc[-tail_rows:].reset_index(drop=True)
    click.secho(f"✅ Loaded {len(df)} rows. Date range: [{df.iloc[0][time_column]} — {df.iloc[-1][time_column]}]", fg="green")

    feature_sets = App.config.get("feature_sets", [])
    if not feature_sets:
        click.secho("⚠️ No feature sets defined in config. Skipping...", fg="yellow")
        return

    click.secho(f"🔬 Generating features using {len(feature_sets)} sets...", fg="cyan")
    all_features = []
    start_time = datetime.now()

    for i, fs in enumerate(feature_sets):
        fs_start = datetime.now()
        click.echo(f"➡️  Feature Set {i+1}/{len(feature_sets)}: {fs.get('generator')}")

        df, new_features = generate_feature_set(df, fs, App.config, App.model_store, last_rows=0)
        all_features.extend(new_features)

        elapsed_fs = datetime.now() - fs_start
        click.secho(f"   ✅ Generated {len(new_features)} features in {str(elapsed_fs).split('.')[0]}", fg="green")

    click.secho(f"🏁 Feature generation complete. Total: {len(all_features)} features", fg="green")
    null_counts = df[all_features].isnull().sum()
    if null_counts.any():
        click.secho("⚠️  Null value summary:", fg="yellow")
        click.echo(null_counts.sort_values(ascending=False))

    out_path = data_path / App.config.get("feature_file_name")
    click.secho(f"💾 Saving feature matrix to {out_path}", fg="cyan")

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        click.secho(f"❌ Unsupported file format: {out_path.suffix}", fg="red")
        return

    txt_path = out_path.with_suffix('.txt')
    with open(txt_path, "a+") as f:
        f.write(", ".join([f'"{feat}"' for feat in all_features]) + "\n\n")

    elapsed = datetime.now() - start_time
    per_feat = elapsed / max(1, len(all_features))
    click.secho(f"✅ Saved {len(df)} records with {len(df.columns)} columns", fg="green")
    click.secho(f"📝 Feature list saved to {txt_path}", fg="blue")
    click.secho(f"⏱️ Finished in {str(elapsed).split('.')[0]} ({str(per_feat).split('.')[0]} per feature)", fg="green")


"""
This command will load a feature file (or any file with close price), and add
top-bot columns according to the label parameter, by finally storing both input
data and the labels in the output file (can be the same file as input).
"""
@generate.command(name="labels", help="🏷️ Generate supervised labels from features and price data.")
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, readable=True),
    required=True,
    help='Path to the configuration file'
)
@click.option('--in-nrows', type=int, default=100_000_000, show_default=True, help="Limit number of rows to load")
@click.option('--tail-rows', type=int, default=0, show_default=True, help="Limit number of rows to process")
def generate_labels(config_file, in_nrows, tail_rows):
    """🏷️ Generate supervised labels from merged features and price data.
    
    Load a file with close price (typically feature matrix),
    compute top-bottom labels, add them to the data, and store to output file.
    """
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.generators import generate_feature_set
    
    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    now = datetime.now()
    time_column = App.config["time_column"]
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("feature_file_name")

    if not file_path.is_file():
        click.secho(f"❌ Data file not found: {file_path}", fg='red')
        return

    click.secho(f"📥 Loading data from {file_path}", fg='cyan')
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=in_nrows)
    else:
        click.secho(f"❌ Unsupported file format: {file_path.suffix}", fg='red')
        return

    if tail_rows:
        df = df.iloc[-tail_rows:]
    df = df.reset_index(drop=True)

    click.secho(f"📊 Loaded {len(df)} records from {df[time_column].iloc[0]} to {df[time_column].iloc[-1]}", fg='cyan')

    label_sets = App.config.get("label_sets", [])
    if not label_sets:
        click.secho("⚠️  No label sets defined in configuration.", fg='yellow')
        return

    click.secho(f"🏷️  Generating labels from {len(label_sets)} label set(s)...", fg='green')
    all_labels = []
    for i, fs in enumerate(label_sets):
        click.echo(f"⚙️  Processing label set {i + 1}/{len(label_sets)} - {fs.get('generator')}")
        df, new_labels = generate_feature_set(df, fs, App.config, App.model_store, last_rows=0)
        all_labels.extend(new_labels)
        click.echo(f"✅ Generated {len(new_labels)} labels.")

    click.echo("\n🧼 NULL value check:")
    click.echo(df[all_labels].isnull().sum().sort_values(ascending=False).to_string())

    out_file_name = App.config.get("matrix_file_name")
    out_path = data_path / out_file_name

    click.secho(f"\n💾 Saving labeled data to {out_path}", fg='blue')
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        click.secho(f"❌ Unknown output format: {out_path.suffix}", fg='red')
        return

    label_txt = out_path.with_suffix('.txt')
    with open(label_txt, "a+") as f:
        f.write(", ".join(f'"{col}"' for col in all_labels) + "\n\n")

    click.secho(f"📝 Stored {len(all_labels)} label names in {label_txt}", fg='blue')

    elapsed = datetime.now() - now
    click.secho(f"\n🏁 Completed in {str(elapsed).split('.')[0]} (avg. per label: {str(elapsed/len(all_labels)).split('.')[0]})", fg='green')


"""
Generate new derived columns according to the signal definitions.
The transformations are applied to the results of ML predictions.
"""
@generate.command(name="signals", help="📈  Derive trade signals from predictions")
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True, readable=True),
    required=True,
    help='Path to the configuration file'
)
@click.option('--in_nrows', type=int, default=100_000_000, help='Max number of rows to load from file')
@click.option('--start_index', type=int, default=0, help='Start index of data to process')
@click.option('--end_index', type=int, default=None, help='End index of data to process')
def generate_signals(config_file, in_nrows, start_index, end_index):
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.common.generators import generate_feature_set

    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    time_column = App.config["time_column"]
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol

    now = datetime.now()

    if not data_path.is_dir():
        click.echo(f"❌ Data folder does not exist: {data_path}")
        return

    # Load predictions file
    file_path = data_path / App.config.get("predict_file_name")
    if not file_path.exists():
        click.echo(f"❌ ERROR: Input file does not exist: {file_path}")
        return

    click.echo(f"📥 Loading predictions from: {file_path}")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=in_nrows)
    else:
        click.echo(f"❌ ERROR: Unsupported file extension '{file_path.suffix}'")
        return

    # Trim and reset
    df = df.iloc[start_index:end_index].reset_index(drop=True)
    click.echo(f"📊 Records: {len(df)} | Time range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    feature_sets = App.config.get("signal_sets", [])
    if not feature_sets:
        click.echo("❌ ERROR: No signal sets defined.")
        return

    click.echo(f"🚀 Generating {len(feature_sets)} signal sets...")

    all_features = []

    for i, fs in enumerate(feature_sets):
        fs_start = datetime.now()
        generator_name = fs.get("generator")
        click.echo(f"➡️  [{i+1}/{len(feature_sets)}] Generator: {generator_name}")

        df, new_features = generate_feature_set(df, fs, App.config, App.model_store, last_rows=0)
        all_features.extend(new_features)

        elapsed = datetime.now() - fs_start
        click.echo(f"✅ Finished generator '{generator_name}' in {str(elapsed).split('.')[0]} — Features: {len(new_features)}")

    click.echo("📈 NULL counts in generated features:")
    click.echo(df[all_features].isnull().sum().sort_values(ascending=False).to_string())

    # Save final data
    out_columns = [col for col in ["time", "open", "high", "low", "close"] if col in df.columns]
    out_columns += App.config.get("labels", []) + all_features
    out_df = df[out_columns]

    out_path = data_path / App.config.get("signal_file_name")
    click.echo(f"💾 Saving signals to: {out_path} ({len(out_df)} rows, {len(out_df.columns)} cols)")

    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        click.echo(f"❌ ERROR: Unsupported output format '{out_path.suffix}'")
        return

    total_time = datetime.now() - now
    click.echo(f"🎉 Signal generation completed in {str(total_time).split('.')[0]}")


# add generate sub-command
sage_ctl.add_command(generate)


"""
Train models for all target labels and all algorithms declared in the configuration using the specified features.
"""
@sage_ctl.command(name="train", help="🧠 Train models on features and labels.")
@click.option('--config-file', '-c', type=click.Path(exists=True, readable=True), required=True,
              help='Path to the configuration file (YAML/TOML/JSON)')
@click.option('--in-nrows', type=int, default=100_000_000,
              help='Maximum number of rows to read from the input file.')
@click.option('--tail-rows', type=int, default=0,
              help='Number of most recent rows to process (for debugging or recency bias).')
@click.option('--store-predictions/--no-store-predictions', default=True,
              help='Toggle to store predictions to disk after training.')
def train_models(config_file: str, in_nrows: int, tail_rows: int, store_predictions: bool):
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.common.generators import train_feature_set

    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    now = datetime.now()
    time_column = App.config["time_column"]
    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("matrix_file_name")

    if not file_path.is_file():
        click.secho(f"❌ Input file not found: {file_path}", fg='red')
        return

    click.secho(f"📥 Loading training data from {file_path}", fg='cyan')
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=in_nrows)
    else:
        click.secho(f"❌ Unsupported input format: {file_path.suffix}", fg='red')
        return

    if tail_rows:
        df = df.iloc[-tail_rows:]
    df = df.reset_index(drop=True)

    click.secho(f"🔍 Dataset: {len(df)} records from {df[time_column].iloc[0]} to {df[time_column].iloc[-1]}", fg='cyan')

    # Extract config values
    label_horizon = App.config["label_horizon"]
    train_length = App.config.get("train_length")
    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    train_feature_sets = App.config.get("train_feature_sets", [])
    algorithms = App.config.get("algorithms", [])

    if not train_feature_sets:
        click.secho("⚠️ No train feature sets defined. Exiting.", fg='yellow')
        return

    # Select only needed columns
    base_cols = [time_column, 'open', 'high', 'low', 'close', 'volume', 'close_time']
    base_cols = [col for col in base_cols if col in df.columns]
    all_features = list(set(train_features + labels))
    df = df[base_cols + [col for col in all_features if col not in base_cols]]

    for label in labels:
        if df[label].dtype == bool:
            df[label] = df[label].astype(int)

    if label_horizon:
        df = df.head(-label_horizon)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=train_features, inplace=True)

    if train_length:
        df = df.tail(train_length)

    if df.empty:
        click.secho("❌ No usable data after filtering NULLs and trimming.", fg='red')
        return

    df.reset_index(drop=True, inplace=True)

    click.secho(f"🚀 Training {len(algorithms)} algorithms on {len(df)} records...", fg='green')

    out_df = pd.DataFrame()
    models = {}
    scores = {}

    for i, fs in enumerate(train_feature_sets):
        click.echo(f"⚙️ Training feature set {i+1}/{len(train_feature_sets)} - Generator: {fs.get('generator')}")
        fs_now = datetime.now()
        fs_out_df, fs_models, fs_scores = train_feature_set(df, fs, App.config)

        out_df = pd.concat([out_df, fs_out_df], axis=1)
        models.update(fs_models)
        scores.update(fs_scores)
        
        fs_elapsed = datetime.now() - fs_now
        click.secho(f"Finished train feature set {i}/{len(train_feature_sets)}. Generator {fs.get('generator')}. Time: {str(fs_elapsed).split('.')[0]}")

    click.secho(f"✅ Finished training models...")
 
    # Store all collected models in files
    for score_column_name, model_pair in models.items():
        App.model_store.put_model_pair(score_column_name, model_pair)

    click.secho(f"✅ Models saved to {App.model_store.model_path.absolute()}", fg='blue')

    # # Store scores
    # lines = list()
    # for score_column_name, score in scores.items():
    #     line = score_column_name + ", " + str(score)
    #     lines.append(line)
        
    # Save metrics
    metrics_path = (data_path / "prediction-metrics.txt").resolve()
    with open(metrics_path, 'a+') as f:
        for name, score in scores.items():
            score = {k: str(v) for k, v in score.items()}
            f.write(f"{name}, {str(score)}\n")
    click.secho(f"📈 Metrics written to {metrics_path}", fg='blue')

    # Optionally save predictions
    if store_predictions:
        predictions_path = data_path / App.config.get("predict_file_name")
        result_df = out_df.join(df[base_cols + labels])

        click.secho(f"💾 Writing predictions to {predictions_path}", fg='cyan')
        if predictions_path.suffix == ".parquet":
            result_df.to_parquet(predictions_path, index=False)
        elif predictions_path.suffix == ".csv":
            result_df.to_csv(predictions_path, index=False, float_format="%.6f")
        else:
            click.secho(f"❌ Unknown file type for predictions: {predictions_path.suffix}", fg='red')
            return
        click.secho(f"📦 Stored {len(result_df)} predictions", fg='green')

    elapsed = datetime.now() - now
    click.secho(f"\n🏁 Finished training in {str(elapsed).split('.')[0]}", fg='green')


"""
Model Predictions
"""
@click.group(name="predict", help="🔮 Predict using trained models (single step or rolling).")
def predict():
    pass


"""
Apply models to (previously generated) features and compute prediction scores.
"""
@predict.command(name="single", help="🔹 Make single predictions using trained models.")
@click.option('--config-file', '-c', type=click.Path(exists=True), required=True, help='Path to configuration file.')
@click.option('--in-nrows', type=int, default=100_000_000, help='Max number of input records to read.')
@click.option('--tail-rows', type=int, default=0, help='Limit to recent rows only.')
def predict_single(config_file, in_nrows, tail_rows):
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.generators import predict_feature_set

    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    time_column = App.config["time_column"]
    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = data_path / App.config.get("matrix_file_name")

    if not file_path.is_file():
        click.echo(f"❌ ERROR: Input file does not exist: {file_path}")
        return

    click.echo(f"📥 Loading data from: {file_path}")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601", nrows=in_nrows)
    else:
        click.echo(f"❌ ERROR: Unsupported file extension '{file_path.suffix}'")
        return

    if tail_rows:
        df = df.iloc[-tail_rows:]

    df = df.reset_index(drop=True)
    click.echo(f"📊 Loaded {len(df)} records. Range: [{df.iloc[0][time_column]}, {df.iloc[-1][time_column]}]")

    train_features = App.config.get("train_features")
    labels = App.config["labels"]
    algorithms = App.config.get("algorithms")

    base_cols = [x for x in ["open", "high", "low", "close", "volume", "close_time", time_column] if x in df.columns]
    labels_present = set(labels).issubset(df.columns)
    selected_features = train_features + labels if labels_present else train_features

    df = df[base_cols + [col for col in selected_features if col not in base_cols]]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=train_features, inplace=True)
    df = df.reset_index(drop=True)

    train_feature_sets = App.config.get("train_feature_sets", [])
    if not train_feature_sets:
        click.echo("❌ ERROR: No train feature sets defined.")
        return

    click.echo(f"🚀 Generating predictions for {len(df)} records...")
    out_df = pd.DataFrame()
    features = []
    scores = {}

    for i, fs in enumerate(train_feature_sets):
        click.echo(f"⚙️  Processing train feature set {i+1}/{len(train_feature_sets)}: {fs.get('generator')}")
        fs_out_df, fs_features, fs_scores = predict_feature_set(df, fs, App.config, App.model_store)
        out_df = pd.concat([out_df, fs_out_df], axis=1)
        features.extend(fs_features)
        scores.update(fs_scores)

    click.echo(f"✅ Finished generating predictions.")

    if labels_present:
        metrics_path = (data_path / "prediction-metrics.txt").resolve()
        with open(metrics_path, 'a+') as f:
            for name, score in scores.items():
                score = {k: str(v) for k, v in score.items()}
                f.write(f"{name}, {score}\n")
        click.echo(f"📈 Metrics written to: {metrics_path}")

    out_df = out_df.join(df[base_cols + (labels if labels_present else [])])
    out_path = data_path / App.config.get("predict_file_name")

    click.echo(f"💾 Saving predictions to: {out_path}")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        click.echo(f"❌ ERROR: Unsupported file format '{out_path.suffix}'")
        return

    click.echo(f"✅ Predictions stored. Rows: {len(out_df)}, Columns: {len(out_df.columns)}")
    click.echo(f"⏱️  Prediction complete in {str(datetime.now() - now).split('.')[0]}")



"""
Generate label predictions for the whole input feature matrix by iteratively training models using historic data and predicting labels for some future horizon.
The main parameter is the step of iteration, that is, the future horizon for prediction.
As usual, we can specify past history length used to train a model.
The output file will store predicted labels in addition to all input columns (generated features and true labels).
This file is intended for training signal models (by simulating trade process and computing overall performance for some long period).
The output predicted labels will cover shorter period of time because we need some relatively long history to train the very first model.
"""
@predict.command(name="multiple", help="🔁 Run rolling window predictions.")
@click.option('--config-file', '-c', type=click.Path(exists=True), required=True, help='Path to configuration file.')
@click.option('--in-nrows', type=int, default=100_000_000, show_default=True, help='Max rows to read from input file.')
@click.option('--max-workers', type=int, default=4, help='Number of parallel workers.')
def predict_rolling(config_file, in_nrows, max_workers):
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.utils import find_index
    from intelligent_trading_bot.common.classifiers import train_predict_gb, train_predict_nn, train_predict_lc, train_predict_svc
    from intelligent_trading_bot.common.classifiers import compute_scores, compute_scores_regression
    from intelligent_trading_bot.common.model_store import ModelStore, label_algo_separator, score_to_label_algo_pair

    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    now = datetime.now()
    cfg = App.config
    rp_cfg = cfg["rolling_predict"]

    time_column = cfg["time_column"]
    symbol = cfg["symbol"]
    data_path = Path(cfg["data_folder"]) / symbol
    file_path = data_path / cfg.get("matrix_file_name")

    if not file_path.exists():
        click.echo(f"❌ File not found: {file_path}")
        return

    click.echo(f"📥 Loading: {file_path}")
    df = (pd.read_parquet(file_path) if file_path.suffix == ".parquet"
          else pd.read_csv(file_path, parse_dates=[time_column], nrows=in_nrows))

    # Slice based on config
    start = find_index(df, rp_cfg["data_start"]) if isinstance(rp_cfg.get("data_start"), str) else rp_cfg.get("data_start", 0)
    end = find_index(df, rp_cfg["data_end"]) if isinstance(rp_cfg.get("data_end"), str) else rp_cfg.get("data_end", None)
    df = df.iloc[start:end].reset_index(drop=True)
    click.echo(f"📊 Records: {len(df)} | Time range: [{df[time_column].iloc[0]}, {df[time_column].iloc[-1]}]")

    # Rolling loop configuration
    start_idx = find_index(df, rp_cfg["prediction_start"]) if isinstance(rp_cfg.get("prediction_start"), str) else rp_cfg.get("prediction_start")
    size = rp_cfg["prediction_size"]
    steps = rp_cfg["prediction_steps"]

    if not start_idx:
        start_idx = len(df) - size * steps
    elif not size:
        size = (len(df) - start_idx) // steps
    elif not steps:
        steps = (len(df) - start_idx) // size

    if len(df) - start_idx < steps * size:
        raise ValueError("❌ Not enough data for rolling loop.")

    # Prep
    label_horizon = cfg["label_horizon"]
    features = cfg["train_features"]
    labels = cfg["labels"]
    algorithms = cfg["algorithms"]
    out_columns = [col for col in ["open", "high", "low", "close", "volume", "close_time", time_column] if col in df.columns]
    df = df[out_columns + [col for col in features + labels if col not in out_columns]].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    use_multiprocessing = rp_cfg.get("use_multiprocessing", False)
    max_workers = rp_cfg.get("max_workers", None)

    # Run rolling
    result_df = pd.DataFrame()
    click.echo(f"🌀 Rolling from idx={start_idx} for {steps} steps of {size} rows each")

    for step in range(steps):
        predict_start = start_idx + step * size
        predict_end = predict_start + size

        predict_df = df.iloc[predict_start:predict_end]
        predict_labels_df = pd.DataFrame(index=predict_df.index)
        df_X_test = predict_df[features]

        train_end = predict_start - label_horizon - 1
        train_start = max(0, train_end - cfg.get("train_length", train_end))
        train_df = df.iloc[train_start:train_end].dropna(subset=features)
        if len(train_df) == 0:
            click.echo(f"⚠️  Skipping step {step+1}/{steps}: No training data available in range [{train_start}:{train_end}]")
            continue


        click.echo(f"\n📌 Step {step+1}/{steps} | Train [{train_start}:{train_end}] | Predict [{predict_start}:{predict_end}]")
        step_start = datetime.now()

        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for label in labels:
                    for model in algorithms:
                        key = f"{label}{label_algo_separator}{model['name']}"
                        df_X = train_df[features]
                        df_y = train_df[label]
                        func = {
                            "gb": train_predict_gb,
                            "nn": train_predict_nn,
                            "lc": train_predict_lc,
                            "svc": train_predict_svc,
                        }.get(model["algo"])

                        if func is None:
                            click.echo(f"❌ Unknown model type: {model['algo']}")
                            return

                        futures[key] = executor.submit(func, df_X, df_y, df_X_test, model)

                for k, f in futures.items():
                    predict_labels_df[k] = f.result()
        else:
            for label in labels:
                for model in algorithms:
                    key = f"{label}{label_algo_separator}{model['name']}"
                    func = {
                        "gb": train_predict_gb,
                        "nn": train_predict_nn,
                        "lc": train_predict_lc,
                        "svc": train_predict_svc,
                    }.get(model["algo"])

                    if func is None:
                        click.echo(f"❌ Unknown model type: {model['algo']}")
                        return

                    df_X = train_df[features]
                    df_y = train_df[label]
                    predict_labels_df[key] = func(df_X, df_y, df_X_test, model)

        result_df = pd.concat([result_df, predict_labels_df])
        click.echo(f"✅ Step {step+1} done in {str(datetime.now() - step_start).split('.')[0]}")

    # Save
    out_df = result_df.join(df[out_columns + labels])
    out_path = data_path / cfg.get("predict_file_name")

    click.echo(f"💾 Saving predictions to {out_path}")
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        out_df.to_csv(out_path, index=False, float_format="%.6f")
    else:
        click.echo(f"❌ Unsupported file format: {out_path.suffix}")
        return

    # Scoring
    score_lines = []
    for col in result_df.columns:
        label, _ = score_to_label_algo_pair(col)
        score_df = out_df[[label, col]].dropna()
        y_true = score_df[label].astype(int)
        y_pred = score_df[col]

        score = (compute_scores_regression(y_true, y_pred) if is_float_dtype(y_true)
                 else compute_scores(y_true, y_pred))

        score_lines.append(f"{col}, AUC={score.get('auc'):.3f}, AP={score.get('ap'):.3f}, F1={score.get('f1'):.3f}")

    with open(out_path.with_suffix('.txt'), "a+") as f:
        f.write("\n".join(score_lines) + "\n\n")

    click.echo(f"🏁 Rolling prediction finished in {str(datetime.now() - now).split('.')[0]}")

sage_ctl.add_command(predict)


"""
The command is intended for finding best trade parameters for a certain trade algorithm
by executing trade simulation (backtesting) for all specified parameters.
It performs exhaustive search in the space of all specified parameters by computing 
trade performance and then choosing the parameters with the highest profit (or maybe
using other selection criteria like stability of the results or minimum allowed losses etc.)

Notes:
- The optimization is based on certain trade algorithm. This means that a trade algorithm
is a parameter for this script. Different trade algorithms have different trade logics and 
also have different parameters. Currently, the script works with a very simple threshold-based
trade algorithm: if some score is higher than the threshold (parameter) then buy, if it is lower
than another threshold then sell. There is also a version with two thresholds for two scores.
- The script consumes the results of signal script but it then varies parameters of one entry
responsible for generation of trade signals. It then measures performance.
"""
@sage_ctl.command(name="simulate", help="🧪  Run signal simulation using exhaustive parameter grid and backtesting")
@click.option('--config_file', '-c', type=click.Path(), default='', help='Path to config file')
def simulate(config_file):
    """Run signal simulation using exhaustive parameter grid and backtesting."""
    from sklearn.model_selection import ParameterGrid
    from intelligent_trading_bot.service.App import App
    from intelligent_trading_bot.common.model_store import ModelStore
    from intelligent_trading_bot.common.backtesting import simulated_trade_performance
    from intelligent_trading_bot.common.generators import generate_feature_set

    config = handle_config(config_file)
    App.config.update(config)
    App.model_store = ModelStore(App.config)
    App.model_store.load_models()

    time_column = App.config["time_column"]
    now = datetime.now()

    symbol = App.config["symbol"]
    data_path = Path(App.config["data_folder"]) / symbol
    out_path = data_path

    if not data_path.exists():
        click.echo(f"❌ Data folder does not exist: {data_path}")
        return

    # Load signal data
    file_path = data_path / App.config.get("signal_file_name")
    if not file_path.exists():
        click.echo(f"❌ Input file does not exist: {file_path}")
        return

    click.echo(f"📥 Loading signals from: {file_path}")
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=[time_column], date_format="ISO8601")
    else:
        click.echo(f"❌ Unsupported extension: {file_path.suffix}")
        return

    click.echo(f"📊 Records: {len(df)} | Columns: {len(df.columns)}")

    simulate_config = App.config["simulate_model"]

    # Filter by time or index
    data_start = simulate_config.get("data_start")
    data_end = simulate_config.get("data_end")
    if data_start:
        df = df[df[time_column] >= data_start] if isinstance(data_start, str) else df.iloc[data_start:]
    if data_end:
        df = df[df[time_column] < data_end] if isinstance(data_end, str) else df.iloc[:-data_end]

    df = df.reset_index(drop=True)
    click.echo(f"📉 Filtered Data: {len(df)} records | Range: [{df[time_column].iloc[0]}, {df[time_column].iloc[-1]}]")

    months_in_simulation = (df[time_column].iloc[-1] - df[time_column].iloc[0]) / timedelta(days=30)

    # Load parameter grid
    parameter_grid = simulate_config["grid"]
    direction = simulate_config.get("direction", "")
    topn_to_store = simulate_config.get("topn_to_store", 10)

    if direction not in ["long", "short"]:
        raise ValueError(f"❌ Invalid direction: {direction}. Use 'long' or 'short'.")

    for k in ["buy_signal_threshold", "sell_signal_threshold", "buy_signal_threshold_2", "sell_signal_threshold_2"]:
        if isinstance(parameter_grid.get(k), str):
            parameter_grid[k] = eval(parameter_grid[k])

    if simulate_config.get("buy_sell_equal"):
        parameter_grid["sell_signal_threshold"] = [None]
        parameter_grid["sell_signal_threshold_2"] = [None]

    generator_name = simulate_config["signal_generator"]
    signal_generator = next((x for x in App.config.get("signal_sets", []) if x.get("generator") == generator_name), None)

    if not signal_generator:
        raise ValueError(f"❌ Signal generator '{generator_name}' not found in signal_sets.")

    click.echo(f"🧪 Simulating parameter grid for generator '{generator_name}'...")

    performances = []
    for parameters in tqdm(ParameterGrid([parameter_grid]), desc="GRID SEARCH"):

        if simulate_config.get("buy_sell_equal"):
            parameters["sell_signal_threshold"] = -parameters["buy_signal_threshold"]
            if parameters.get("buy_signal_threshold_2"):
                parameters["sell_signal_threshold_2"] = -parameters["buy_signal_threshold_2"]

        signal_generator["config"]["parameters"].update(parameters)
        df, _ = generate_feature_set(df, signal_generator, App.config, App.model_store, last_rows=0)

        buy_col, sell_col = signal_generator["config"]["names"][:2]

        perf, perf_long, perf_short = simulated_trade_performance(df, buy_col, sell_col, 'close')

        use_perf = perf_long if direction == "long" else perf_short
        use_perf["#transactions/M"] = round(use_perf["#transactions"] / months_in_simulation, 2)
        use_perf["profit/M"] = round(use_perf["profit"] / months_in_simulation, 2)
        use_perf["%profit/M"] = round(use_perf["%profit"] / months_in_simulation, 2)

        performances.append({
            "model": parameters,
            "performance": use_perf
        })

    # Sort by profit per month
    performances = sorted(performances, key=lambda x: x["performance"]["%profit/M"], reverse=True)[:topn_to_store]

    # Flatten and write output
    out_file = (out_path / App.config.get("signal_models_file_name")).with_suffix(".txt")
    is_new = not out_file.exists()
    keys = list(performances[0]["model"].keys()) + list(performances[0]["performance"].keys())
    lines = []

    for p in performances:
        line = list(p["model"].values()) + list(p["performance"].values())
        lines.append(",".join(str(x) for x in line))

    with open(out_file, "a+") as f:
        if is_new:
            f.write(",".join(keys) + "\n")
        f.write("\n".join(lines) + "\n\n")

    click.echo(f"✅ Simulation results stored in: {out_file} ({len(lines)} lines)")
    click.echo(f"⏱️  Finished simulation in {str(datetime.now() - now).split('.')[0]}")


"""
    Execute trades in live/paper mode
"""
# @sage_ctl.command(name="live", help="💸  Run Sage in live prediction mode")
# @click.option(
#     '--config-file', '-c',
#     type=click.Path(exists=True, readable=True),
#     required=True,
#     help='Path to the configuration file'
# )
# def live(config_file):
#     """
#     Run Sage in live prediction mode with signal + (optional) trade.
#     """
#     from intelligent_trading_bot.service.App import App
#     from intelligent_trading_bot.pipline.live import LiveTradingEngine
    
#     config = handle_config(config_file)
#     App.config.update(config)
#     engine = LiveTradingEngine(App)
#     engine.run()


"""
Async Bus
"""
@sage_ctl.command("bus-summary")
@click.option("--topic", type=str, help="Filter by topic name.")
@click.option("--stats", is_flag=True, help="Include runtime stats like read/write indices.")
@click.option("--json", "as_json", is_flag=True, help="Output summary as JSON.")
def bus_summary(topic, stats, as_json):
    """Show summary of all registered channels (buses)."""
    registry = ChannelRegistry.global_instance()
    buses = registry.all_buses()

    filtered = {k: v for k, v in buses.items() if topic is None or topic in k}
    summaries = {}

    for name, channel in filtered.items():
        bus = AsyncMappedBus(name, channel.slot_size, channel.num_slots, channel.max_consumers)
        summaries[name] = bus.summary(stats=stats, as_dict=as_json)

    if as_json:
        click.echo(json.dumps(summaries, indent=2))
    else:
        for name, info in summaries.items():
            click.echo(f"\n{name}")
            click.echo("-" * len(name))
            click.echo(info)


@sage_ctl.command("bus-memory")
def bus_memory():
    """Display memory usage estimates for all buses."""
    registry = ChannelRegistry.global_instance()
    buses = registry.all_buses()

    for name, channel in buses.items():
        size = channel.slot_size * channel.num_slots
        size_kb = size / 1024
        click.echo(f"{name}: {size_kb:.2f} KB")


@sage_ctl.command("bus-inspect")
@click.argument("topic")
@click.option("--consumer", type=int, default=0, help="Consumer ID to inspect.")
def bus_inspect(topic, consumer):
    """Inspect a specific bus/channel."""
    registry = ChannelRegistry.global_instance()
    if not registry.exists(topic):
        click.echo(f"Channel '{topic}' not found.")
        return

    channel = registry.get(topic)
    bus = AsyncMappedBus(topic, channel.slot_size, channel.num_slots, channel.max_consumers)
    summary = bus.summary(stats=True, as_dict=True)

    click.echo(json.dumps(summary, indent=2))
