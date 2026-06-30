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
import os
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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