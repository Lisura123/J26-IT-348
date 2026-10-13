from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


@lru_cache
def users() -> pd.DataFrame:
    return pd.read_csv(DATA / "dashboard_users.csv")


@lru_cache
def permissions() -> dict[str, set[str]]:
    """{'ADMIN': {'view_shap', ...}, 'MANAGER': {...}} - only permissions with allowed=True."""
    p = pd.read_csv(DATA / "roles_permissions.csv")
    p = p[p.allowed.astype(str).str.lower() == "true"]
    return p.groupby("role").permission.apply(set).to_dict()