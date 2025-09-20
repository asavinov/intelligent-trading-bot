"""
Pytest bootstrap: ensure the project root is on sys.path so local packages
like 'common' can be imported when running tests from the repository root.
"""
from pathlib import Path
import sys


def _ensure_project_root_on_path() -> None:
    # tests/ is inside the project root; add its parent to sys.path if missing
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()
