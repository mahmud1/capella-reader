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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from osgeo import gdal

logger = logging.getLogger(__name__)

gdal.UseExceptions()
C_M_PER_S = 299_792_458.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlcRecord:
    """Container for one SLC path and its decoded Capella metadata."""

    path: Path
    center_time: datetime
    metadata: dict[str, Any]
    date: str = field(init=False)
    length: float = field(init=False)
    width: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", self.center_time.strftime("%Y%m%d"))

        ds = open_gdal(self.path)
        object.__setattr__(self, "length", int(ds.RasterYSize))
        object.__setattr__(self, "width", int(ds.RasterXSize))


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


def discover_slcs(paths: Sequence[Path], sort_by_time: bool = True) -> list[SlcRecord]:
    """Read metadata for every SLC path and return validated/sorted records."""

    records = []
    for path in paths:
        meta = load_capella_metadata(path)
        center = datetime_from_meta(meta)
        records.append(SlcRecord(path=path, center_time=center, metadata=meta))
    if sort_by_time:
        records.sort(key=lambda r: (r.center_time, str(r.path)))

    # TODO: check duplicate acquisitions based on center_time?
    return records


def read_slc_array(path: Path) -> np.ndarray:
    """Read one Capella SLC TIFF as complex64."""

    ds = open_gdal(path)
    arr = ds.ReadAsArray()

    if arr is None:
        raise ValueError(f"Failed to read raster data from {path}")

    arr = np.asarray(arr)

    if not np.iscomplexobj(arr):
        raise TypeError(f"Raster data in {path} is not complex.")

    if arr.ndim != 2:
        raise ValueError(f"Raster data in {path} has {arr.ndim} dimensions; expected 2.")

    return arr.astype(np.complex64, copy=False)


def validate_stack_dimensions(records: Sequence[SlcRecord]) -> None:
    """Validate that all SLCs share the same dimension."""

    length, width = records[0].length, records[0].width
    for rec in records[1:]:
        shp = (rec.length, rec.width)
        if shp != (length, width):
            raise ValueError(
                f"SLC shape mismatch: {rec.path} has shape {shp}, expected {(length, width)}"
            )
    return


# ---------------------------------------------------------------------------
# Attributes helpers
# ---------------------------------------------------------------------------


def build_common_attrs(record: SlcRecord) -> dict[str, Any]:
    """Create common MiaplPy root attributes."""

    width = record.width
    length = record.length

    meta = record.metadata
    image = nested_get(meta, ["collect", "image"], {}) or {}
    radar = nested_get(meta, ["collect", "radar"], {}) or {}

    alooks = image.get("azimuth_looks", 1)
    azimuth_pixel_spacing = image.get("azimuth_resolution", "")

    # Antenna side follows the historical ROI_PAC metadata convention:
    # right-looking: -1. left looking: +1.
    pointing = radar.get("pointing", None)
    antenna_side = 1 if pointing == "left" else -1 if pointing == "right" else None
    orbit_direction = nested_get(meta, ["collect", "state", "direction"], None)

    heading_angle = compute_heading(meta)

    platform = nested_get(meta, ["collect", "platform"], None)

    transmit_polarization = radar.get("transmit_polarization")
    receive_polarization = radar.get("receive_polarization")
    if transmit_polarization is None or receive_polarization is None:
        polarization = None
    else:
        polarization = f"{transmit_polarization}{receive_polarization}"

    range_pixel_size = image.get("range_resolution")

    rlooks = image.get("range_looks", 1)

    starting_range = nested_get(image, ["image_geometry", "range_to_first_sample"])

    freq = radar.get("center_frequency")
    wavelength = C_M_PER_S / float(freq) if freq else None

    attrs: dict[str, Any] = {
        "ALOOKS": alooks,
        "ANTENNA_SIDE": antenna_side,
        "AZIMUTH_PIXEL_SIZE": azimuth_pixel_spacing,
        "DATA_TYPE": "complex64",
        "FILE_LENGTH": length,
        "FILE_TYPE": "slc",
        "HEADING": heading_angle,
        "LENGTH": length,
        "ORBIT_DIRECTION": orbit_direction,
        "PLATFORM": platform,
        "POLARIZATION": polarization,
        "PROCESSOR": "capella-reader",
        "RANGE_PIXEL_SIZE": range_pixel_size,
        "RLOOKS": rlooks,
        "STARTING_RANGE": starting_range,
        "WAVELENGTH": wavelength,
        "WIDTH": width,
        "XMAX": width - 1,
        "YMAX": length - 1,
    }

    return attrs


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def compute_heading(meta: Mapping[str, Any]) -> float:
    """
    Compute the approximate satellite heading angle (degrees clockwise from north)
    at the image center.

    The heading is derived from the orbit velocity nearest to
    ``collect.image.center_pixel.center_time``. The velocity vector is projected
    onto the local east-north-up (ENU) frame at
    ``collect.image.center_pixel.target_position``, and the heading is measured
    clockwise from geographic north.
    """
    target = _vec(nested_get(meta, ["collect", "image", "center_pixel", "target_position"]))
    vel = _velocity_at_center_time(meta)

    # approximate geocentric latitude / longitude
    x, y, z = target
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y))

    east = np.array([
        -np.sin(lon),
         np.cos(lon),
         0.0,
    ])

    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat),
    ])

    v_e = np.dot(vel, east)
    v_n = np.dot(vel, north)

    heading = np.degrees(np.arctan2(v_e, v_n))
    return (heading + 360.0) % 360.0

# ---------------------------------------------------------------------------
# Perpendicular Baseline
# ---------------------------------------------------------------------------


def _vec(value: Any) -> np.ndarray:
    """Convert Capella vector dict/list into a float64 vector."""

    if isinstance(value, Mapping):
        return np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    """Return unit vector, raising on invalid/zero norm."""

    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(f"Invalid zero-length vector for {name}")
    return v / n


def _state_vectors(meta: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Return Capella orbit state vectors from the known metadata path."""

    states = nested_get(meta, ["collect", "state", "state_vectors"])
    if not states:
        raise ValueError("Could not find collect.state.state_vectors in Capella metadata")

    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise TypeError("collect.state.state_vectors must be a sequence")

    for i, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise TypeError(f"collect.state.state_vectors[{i}] must be a mapping")
        if "time" not in state or "position" not in state or "velocity" not in state:
            raise ValueError(
                f"collect.state.state_vectors[{i}] must contain time, position, and velocity"
            )

    return states


def _velocity_at_center_time(meta: Mapping[str, Any]) -> np.ndarray:
    """Return orbit velocity nearest to collect.image.center_pixel.center_time."""

    center_time = nested_get(meta, ["collect", "image", "center_pixel", "center_time"])
    if center_time is None:
        raise ValueError("Could not find collect.image.center_pixel.center_time in metadata")

    t0 = parse_iso8601(str(center_time))
    states = _state_vectors(meta)

    state = min(
        states,
        key=lambda s: abs((parse_iso8601(str(s["time"])) - t0).total_seconds()),
    )

    vel = state["velocity"]
    return np.array([vel["vx"], vel["vy"], vel["vz"]], dtype=np.float64)


def compute_bperp(records: Sequence[SlcRecord], ref_index: int = 0, flip_sign: bool = False) -> np.ndarray:
    """
    Compute signed approximate perpendicular baselines relative to one reference acquisition.

    The baseline is computed from Capella ECEF metadata at the reference scene geometry.
    The reference satellite position and reference target position define the line-of-sight
    vector, while the reference orbit velocity defines the along-track direction. The
    signed perpendicular direction is the orbital-frame cross-track direction:

        los = unit(target_ref - sat_ref)
        along_track = unit(reference_velocity)
        along_track = unit(along_track - dot(along_track, los) * los)
        cross_track = unit(cross(along_track, los))
        bperp_i = dot(sat_i - sat_ref, cross_track)

    This produces a signed Bperp value for each acquisition in meters. The reference
    acquisition has Bperp close to zero by construction. The sign follows the cross-track
    orientation defined by cross(along_track, los). Set flip_sign=True if an external
    processor shows the opposite convention.

    """
    if not records:
        return np.asarray([], dtype=np.float32)

    metas = [r.metadata for r in records]
    ref_meta = metas[ref_index]

    sat_ref = _vec(nested_get(ref_meta, ["collect", "image", "reference_antenna_position"]))
    tgt_ref = _vec(nested_get(ref_meta, ["collect", "image", "reference_target_position"]))
    vel_ref = _velocity_at_center_time(ref_meta)

    los = _unit(tgt_ref - sat_ref, "reference LOS")
    along_track = _unit(vel_ref, "reference velocity")

    # Keep only the component of velocity perpendicular to LOS.
    along_track = _unit(
        along_track - np.dot(along_track, los) * los,
        "LOS-orthogonal reference velocity",
    )

    cross_track = _unit(np.cross(along_track, los), "cross-track direction")
    if flip_sign:
        cross_track *= -1.0

    out: list[float] = []
    for meta in metas:
        sat = _vec(nested_get(meta, ["collect", "image", "reference_antenna_position"]))
        baseline = sat - sat_ref

        bperp = float(np.dot(baseline, cross_track))

        out.append(bperp)

    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# HDF5 writers
# ---------------------------------------------------------------------------


def write_slc_stack(
    out_file: Path,
    records: Sequence[SlcRecord],
    compression: str | None,
) -> None:
    """Write MiaplPy-style slcStack.h5."""

    length, width = records[0].length, records[0].width
    out_file.parent.mkdir(parents=True, exist_ok=True)
    attrs = build_common_attrs(records[0])
    attrs.update(
        {
            "FILE_PATH": str(out_file),
        }
    )

    with h5py.File(out_file, "w") as f:
        write_attrs(f, attrs)
        dset = f.create_dataset(
            "slc",
            shape=(len(records), length, width),
            dtype=np.complex64,
            compression=compression,
        )
        f.create_dataset(
            "date",
            data=np.asarray([r.date.encode("ascii") for r in records], dtype="|S8"),
        )

        f.create_dataset(
            "acquisition_time",
            data=np.asarray(
                [r.center_time.isoformat().encode("ascii") for r in records],
                dtype="S32",
            ),
        )

        for i, rec in enumerate(records):
            logger.info(f"[{i + 1}/{len(records)}] writing SLC {rec.date}: {rec.path}")
            arr = read_slc_array(rec.path)
            if arr.shape != (length, width):
                raise ValueError(
                    f"SLC array shape mismatch after read: {rec.path} has {arr.shape}, "
                    f"expected {(length, width)}"
                )
            dset[i, :, :] = arr
        f.create_dataset(
            "bperp",
            data=compute_bperp(records, ref_index=0),
            dtype=np.float32,
        )



def write_attrs(h5obj: Any, attrs: Mapping[str, Any]) -> None:
    """Write scalar/string HDF5 attributes."""

    for key, value in attrs.items():
        if value is None:
            continue
        h5obj.attrs[str(key)] = value


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
        "--out-dir",
        type=Path,
        default=Path("inputs"),
        help="Output directory for MiaplPy/SARvey inputs",
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
    compression: str | None = None if args.compression == "none" else args.compression
    slc_paths = read_path_list(args.slc_list)
    records = discover_slcs(slc_paths, sort_by_time=True)
    validate_stack_dimensions(records)
    logger.info(f"Identified {len(records)} SLC(c) with {records[0].length}x{records[0].width} dimensions")

    out_dir = args.out_dir.resolve()
    write_slc_stack(
        out_dir / "slcStack.h5",
        records,
        compression=compression,
    )


if __name__ == "__main__":
    main()