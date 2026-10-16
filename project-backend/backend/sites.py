"""Reference site registry loaded from reference/sites.csv.
Provides canonical lookups for site_id, device_id, panel_id, and protocol.
"""
import csv
from pathlib import Path
from typing import Dict, Optional, Set

DEFAULT_SITES_CSV = Path(__file__).resolve().parent.parent / "reference" / "sites.csv"

_SITES: Dict[str, Dict[str, str]] = {}


def load_sites(csv_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Loads sites.csv into an in-memory dictionary keyed by site_id."""
    path = csv_path or DEFAULT_SITES_CSV
    if not path.exists():
        raise FileNotFoundError(f"Sites reference file not found: {path}")

    sites: Dict[str, Dict[str, str]] = {}
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("site_id") or "").strip()
            if sid:
                sites[sid] = {k.strip(): (v.strip() if v else "") for k, v in row.items()}

    global _SITES
    _SITES = sites
    return sites


def get_all_sites() -> Dict[str, Dict[str, str]]:
    """Returns the cached sites map, loading from DEFAULT_SITES_CSV if needed."""
    if not _SITES:
        load_sites()
    return _SITES


def valid_site_ids() -> Set[str]:
    """Returns the set of all valid canonical site IDs."""
    return set(get_all_sites().keys())


def get_site(site_id: str) -> Optional[Dict[str, str]]:
    """Returns the site metadata dictionary for a given site_id, or None if unknown."""
    return get_all_sites().get(site_id)


def get_device_id(site_id: str) -> Optional[str]:
    """Returns the canonical device_id for a given site_id."""
    site = get_site(site_id)
    return site.get("device_id") if site else None


def get_protocol(site_id: str) -> Optional[str]:
    """Returns the canonical protocol for a given site_id."""
    site = get_site(site_id)
    return site.get("protocol") if site else None


def get_panel_id(site_id: str) -> Optional[str]:
    """Returns the canonical panel_id for a given site_id."""
    site = get_site(site_id)
    return site.get("panel_id") if site else None
