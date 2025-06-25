import json
import click
import click
import pandas as pd
from datetime import datetime
from pathlib import Path
from intelligent_trading_bot.config import handle_config
from intelligent_trading_bot.service.App import App, _merge_data_sources



@click.group()
def sage_cli():
    """
    📊 Sage CLI — Intelligent Trading Automation

    A unified interface for managing data, ML models/pipelines, and automated trading operations.

    🛠️  Core Commands: \n
    ──────────────────────────────────────────────\n
    ⚙️  sage config     → Load and inspect configuration files \n
    🔗  sage merge      → Merge multiple data sources \n
    🧠  sage generate   → Generate features and labels \n
    🏋️  sage train      → Train ML models \n
    🤖  sage predict    → Run predictions (including rolling forecasts) \n
    📈  sage signals    → Derive trade signals from predictions \n
    🧪  sage simulate   → Simulate/backtest strategies \n
    💸  sage live       → Execute trades in live/paper mode \n
    """
    pass


@sage_cli.command(name="config")
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

@sage_cli.command(name="merge")
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

    df_merged = _merge_data_sources(data_sources)

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

