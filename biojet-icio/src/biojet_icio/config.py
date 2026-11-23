"""Load run settings and the parameter / concordance tables in ``config/``."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class Config:
    settings: dict
    root: Path = REPO_ROOT
    tables: dict = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    def path(self, key: str) -> Path:
        return self.root / self.settings["paths"][key]

    @property
    def year(self) -> int:
        return int(self.settings["base"]["year"])

    @property
    def dis(self) -> dict:
        return self.settings["disaggregation"]

    def table(self, name: str) -> pd.DataFrame:
        return self.tables[name]


def load_config(settings_file: str | Path | None = None, root: Path | None = None) -> Config:
    root = Path(root) if root else REPO_ROOT
    cfg_dir = root / "config"
    settings_file = Path(settings_file) if settings_file else cfg_dir / "settings.toml"
    with open(settings_file, "rb") as fh:
        settings = tomllib.load(fh)
    with open(cfg_dir / "user_groups.toml", "rb") as fh:
        user_groups = tomllib.load(fh)

    read = lambda f, **kw: pd.read_csv(cfg_dir / f, **kw)  # noqa: E731
    tables = {
        "regions": read("icio_regions.csv", dtype={"comtrade_code": int}),
        "biofuels": read("biofuels.csv").set_index("code"),
        "hs": read("hs_biofuel_concordance.csv", dtype={"hs6": str}),
        "cost": read("cost_structures.csv"),
        "use": read("use_allocation.csv"),
        "parents": read("biofuel_parent_overrides.csv"),
        "crops": read("feedstock_crops.csv", dtype={"hs_codes": str}),
        "emission_factors": read("emission_factors.csv").set_index("parameter"),
        "special_partners": read("comtrade_special_partners.csv"),
        "user_groups": user_groups,
    }
    return Config(settings=settings, root=root, tables=tables)
