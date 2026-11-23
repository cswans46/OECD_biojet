"""Download the public inputs (run on a machine with internet access).

    python scripts/download_data.py --unsd --faostat --comtrade --oecd-ghg

* --comtrade  UN Comtrade bilateral exports and imports, all reporters, for the
              biofuel HS codes (and crop HS codes with --crops). A free
              subscription key (https://comtradeplus.un.org) in the
              COMTRADE_KEY env var lifts the anonymous preview limits;
              without one the script uses the public preview endpoint, one
              reporter per call, with pauses.
* --unsd      UNSD energy statistics (UNdata EDATA) for biogasoline (AL),
              biodiesel (BD), bio jet kerosene (BJ), other liquid biofuels (OL).
* --faostat   FAOSTAT bulk files: Bioenergy (BE), Crops & livestock (QCL),
              Value of agricultural production (QV).
* --oecd-ghg  OECD emissions by ICIO industry via SDMX (set --ghg-flow/--ghg-key).
* --icio      OECD ICIO 2025 zip (pass --icio-url; the OECD page lists the links).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))
from biojet_icio.config import load_config  # noqa: E402

FAOSTAT = {
    "BE": "https://bulks-faostat.fao.org/production/Environment_Bioenergy_E_All_Data_(Normalized).zip",
    "QCL": "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    "QV": "https://bulks-faostat.fao.org/production/Value_of_Production_E_All_Data_(Normalized).zip",
}
UNDATA = "https://data.un.org/Handlers/DownloadHandler.ashx?DataFilter=cmID:{cm};yr:{yr}&DataMartId=EDATA&Format=csv"
COMTRADE_GET = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
COMTRADE_PREVIEW = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"


def _unzip_to(content: bytes, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for n in z.namelist():
            if n.lower().endswith(".csv"):
                p = dest / Path(n).name
                p.write_bytes(z.read(n))
                out.append(p)
    return out


def get_unsd(year: int) -> None:
    for cm in ("AL", "BD", "BJ", "OL"):
        r = requests.get(UNDATA.format(cm=cm, yr=year), timeout=120)
        r.raise_for_status()
        for p in _unzip_to(r.content, RAW / "unsd_energy"):
            p.rename(p.with_name(f"UNdata_Export_{cm}_{year}.csv"))
        print("UNSD", cm, "ok")


def get_faostat(domains=("BE", "QCL", "QV")) -> None:
    for d in domains:
        r = requests.get(FAOSTAT[d], timeout=600)
        r.raise_for_status()
        files = _unzip_to(r.content, RAW / "faostat")
        print("FAOSTAT", d, [f.name for f in files])


def get_comtrade(year: int, hs_codes: list[str], reporters: list[int], flows=("X", "M"),
                 pause: float = 7.0) -> None:
    key = os.environ.get("COMTRADE_KEY")
    out = RAW / "comtrade"
    out.mkdir(parents=True, exist_ok=True)
    for hs in hs_codes:
        for fl in flows:
            frames = []
            if key:
                r = requests.get(COMTRADE_GET, params={"period": year, "cmdCode": hs, "flowCode": fl,
                                                       "partner2Code": 0, "customsCode": "C00", "motCode": 0},
                                 headers={"Ocp-Apim-Subscription-Key": key}, timeout=300)
                r.raise_for_status()
                frames.append(pd.DataFrame(r.json().get("data", [])))
            else:
                for rep in reporters:
                    for attempt in range(3):
                        r = requests.get(COMTRADE_PREVIEW, params={"reporterCode": rep, "period": year,
                                                                   "cmdCode": hs, "flowCode": fl}, timeout=120)
                        if r.status_code == 429:
                            time.sleep(60)
                            continue
                        break
                    if r.ok:
                        frames.append(pd.DataFrame(r.json().get("data", [])))
                    time.sleep(pause)
            if frames:
                d = pd.concat(frames, ignore_index=True)
                d.to_csv(out / f"comtrade_{hs}_{fl}_{year}.csv", index=False)
                print("Comtrade", hs, fl, len(d), "records")


def get_oecd_ghg(flow: str, key: str, year: int) -> None:
    url = f"https://sdmx.oecd.org/public/rest/data/{flow}/{key}"
    r = requests.get(url, params={"startPeriod": year, "endPeriod": year, "format": "csvfilewithlabels"},
                     timeout=600)
    r.raise_for_status()
    dest = RAW / "oecd_ghg"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"emissions_by_industry_{year}.csv").write_bytes(r.content)
    print("OECD GHG ok")


def get_icio(url: str) -> None:
    r = requests.get(url, timeout=3600)
    r.raise_for_status()
    files = _unzip_to(r.content, ROOT / "data" / "icio")
    print("ICIO files:", [f.name for f in files])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--unsd", action="store_true")
    ap.add_argument("--faostat", action="store_true")
    ap.add_argument("--comtrade", action="store_true")
    ap.add_argument("--crops", action="store_true", help="add feedstock-crop HS codes to --comtrade")
    ap.add_argument("--oecd-ghg", action="store_true")
    ap.add_argument("--ghg-flow", default="OECD.STI.PIE,DSD_ICIO_GHG_MAIN_2025@DF_ICIO_GHG_MAIN_2025,1.0",
                    help="SDMX dataflow (check the OECD Data Explorer for the emissions-by-industry flow)")
    ap.add_argument("--ghg-key", default="all")
    ap.add_argument("--icio", action="store_true")
    ap.add_argument("--icio-url", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.unsd:
        get_unsd(a.year)
    if a.faostat:
        get_faostat()
    if a.comtrade:
        hs = list(cfg.table("hs").hs6)
        if a.crops:
            for codes in cfg.table("crops").hs_codes:
                hs += [c.zfill(6) if len(c) > 4 else c for c in str(codes).split(";")]
        reps = list(cfg.table("regions").comtrade_code)
        get_comtrade(a.year, hs, reps)
    if a.oecd_ghg:
        get_oecd_ghg(a.ghg_flow, a.ghg_key, a.year)
    if a.icio:
        if not a.icio_url:
            sys.exit("Pass --icio-url (zip link from https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html)")
        get_icio(a.icio_url)


if __name__ == "__main__":
    main()
