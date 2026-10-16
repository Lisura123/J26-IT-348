"""Point specification dataclass for representing telemetry points.
Pure Python dataclass without external database dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class PointSpec:
    measurement: str
    time: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)
