#!/usr/bin/env python
"""Convert Capella SLC stacks and radar-coordinate geometry TIFFs
into MiaplPy/SARvey inputs:

    inputs/slcStack.h5
    inputs/geometryRadar.h5

The converter expects all SLCs and geometry rasters to be already coregistered in the
same radar grid.

Usage
-----
    # TODO
"""

import argparse
import json
import os
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from osgeo import gdal

logger = logging.getLogger(__name__)

gdal.UseExceptions()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlcRecord:
    """Container for one SLC path and its decoded Capella metadata."""

    path: Path
    date: str
    center_time: datetime
    metadata: dict[str, Any]


def read_path_list(list_file: Path) -> list[Path]:
    """
    Read an ASCII list of SLC TIFF paths.

    Blank lines and lines starting with '#' are ignored. Inline comments are allowed
    only when preceded by whitespace, e.g. '/path/a.tif  # comment'.
    """

    paths: list[Path] = []
    base = list_file.resolve().parent
    for raw in list_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        p = Path(os.path.expandvars(os.path.expanduser(line)))
        if not p.is_absolute():
            p = base / p
        paths.append(p.resolve())
    if not paths:
        raise FileNotFoundError(f"No paths found in {list_file}")
    logger.debug(f"Found {len(paths)} SLC paths in {list_file}")
    return paths


def open_gdal(path: Path) -> gdal.Dataset:
    """Open a file with GDAL."""

    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open {path}")
    return ds


def load_capella_metadata(path: Path) -> dict[str, Any]:
    """Decode Capella JSON/Python-literal metadata from TIFFTAG_IMAGEDESCRIPTION."""

    ds = open_gdal(path)
    desc = ds.GetMetadataItem("TIFFTAG_IMAGEDESCRIPTION")

    try:
        obj = json.loads(desc)
    except json.JSONDecodeError:
        logger.exception("Failed to parse metadata from %s", path)
        raise

    return obj


def nested_get(obj: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Read nested dictionary keys, returning default when any key is absent."""

    cur: Any = obj
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_iso8601(value: str) -> datetime:
    """Parse Capella timestamps in ISO8601 format."""

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Python datetime supports microseconds, not nanoseconds.
    text = re.sub(r"(\.\d{6})\d+(?=[+-]\d\d:?\d\d$)", r"\1", text)
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    return datetime.fromisoformat(text)


def datetime_from_meta(meta: Mapping[str, Any]) -> datetime:
    """Extract center time from Capella metadata."""

    t = nested_get(meta, ["collect", "image", "center_pixel", "center_time"])
    if not t:
        raise ValueError("Could not find center time in Capella metadata")
    dt = parse_iso8601(str(t))
    return dt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Capella Coregistered SLC TIFF stack + radar geometry TIFFs to MiaplPy/SARvey inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--slc-list",
        required=True,
        type=Path,
        help="ASCII file containing one SLC TIFF path per line",
    )
    parser.add_argument(
        "--compression",
        default="none",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write log messages to this file.",
    )
    args = parser.parse_args()

    # setup logger
    console_handler = logging.StreamHandler()

    handlers = [console_handler]

    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    slc_paths = read_path_list(args.slc_list)


if __name__ == "__main__":
    main()