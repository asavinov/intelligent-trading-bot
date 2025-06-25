import os
import json
import yaml
import toml
import numpy as np
import configparser
from pathlib import Path
from typing import Union


def handle_config(path: Union[str, Path], mode="load", data=None):
    """
    Loads or saves configuration data in multiple formats based on file extension.

    ## Supported Formats:
    - `.json`   → JSON config
    - `.yaml`, `.yml` → YAML config
    - `.toml`   → TOML config
    - `.ini`    → INI-style config with sections
    - `.npy`    → NumPy serialized dictionary (object arrays allowed)

    ## Parameters:
    - path (str): Path to the config file (input or output).
    - mode (str): Either `"load"` to read or `"save"` to write the config.
    - data (dict/any): The data to save. Required only for mode `"save"`.

    ## Returns:
    - dict or object: Parsed config when `mode="load"`. Nothing when saving.

    ## Usage Examples:
    ```python
    # Load from various formats
    cfg = handle_config("settings.toml", mode="load")
    cfg = handle_config("params.yaml", mode="load")

    # Save to formats
    handle_config("backup.json", mode="save", data=cfg)
    handle_config("params.ini", mode="save", data={"model": {"alpha": "0.5", "beta": "1.0"}})
    ```

    ## Raises:
    - ValueError: If file extension is unsupported or saving without data.
    """

    ext = os.path.splitext(path)[-1].lower()

    if mode == "load":
        if ext == ".json":
            with open(path, "r") as f:
                # Remove everything starting with // and till the line end
                # conf_str = re.sub(r"//.*$", "", conf_str, flags=re.M)
                return json.load(f)

        elif ext in [".yaml", ".yml"]:
            with open(path, "r") as f:
                return yaml.safe_load(f)

        elif ext == ".toml":
            return toml.load(path)

        elif ext == ".ini":
            parser = configparser.ConfigParser()
            parser.read(path)
            return {section: dict(parser.items(section)) for section in parser.sections()}

        elif ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            return arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object else arr

        else:
            raise ValueError(f"Unsupported file type for loading: {ext}")

    elif mode == "save":
        if data is None:
            raise ValueError("Must provide data when saving.")

        if ext == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        elif ext in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.safe_dump(data, f)

        elif ext == ".toml":
            with open(path, "w") as f:
                toml.dump(data, f)

        elif ext == ".ini":
            parser = configparser.ConfigParser()
            for section, params in data.items():
                parser[section] = params
            with open(path, "w") as f:
                parser.write(f)

        elif ext == ".npy":
            np.save(path, data, allow_pickle=True)

        else:
            raise ValueError(f"Unsupported file type for saving: {ext}")

    else:
        raise ValueError("Mode must be 'load' or 'save'")

