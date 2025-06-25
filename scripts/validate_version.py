import re
import tomllib  # Python 3.11+
from pathlib import Path

# Load version from __init__.py
version_file = Path("intelligent_trading_bot/__init__.py")
version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', version_file.read_text())
py_version = version_match.group(1) if version_match else None

# Load version from pyproject.toml
pyproject = tomllib.loads(Path("pyproject.toml").read_text())
toml_version = pyproject["project"]["version"]

if py_version != toml_version:
    print(f"❌ Version mismatch: __init__.py = {py_version}, pyproject.toml = {toml_version}")
    exit(1)

print(f"✅ Version match: {py_version}")
