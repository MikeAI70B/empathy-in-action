import os
from typing import Dict, Optional


def _default_keys_path() -> str:
    # Try CWD first
    cwd_path = os.path.abspath(os.path.join(os.getcwd(), ".PROVIDER_KEYS"))
    if os.path.exists(cwd_path):
        return cwd_path
    # Then project root relative to this file: empathy/core/ -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root_path = os.path.join(project_root, ".PROVIDER_KEYS")
    return root_path


def load_provider_keys(filepath: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """Load provider API keys from a simple KEY=VALUE file into the environment.

    - filepath: path to the keys file; if None, searches CWD then project root.
    - override: if True, overwrite existing environment variables; otherwise only set missing ones.
    Returns a dict of keys that were set (masked values not returned for security).
    """
    path = filepath or _default_keys_path()
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"\'")
                if key and (override or key not in os.environ):
                    os.environ[key] = value
                    loaded[key] = "***"
    except Exception:
        print("Warning: provider keys not loaded")
        return loaded
    return loaded


