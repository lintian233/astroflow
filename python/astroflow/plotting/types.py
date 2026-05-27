from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CandidateInfo:
    dm: float
    toa: float
    freq_start: float
    freq_end: float
    dmt_idx: int
    ref_toa: float
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None

    @classmethod
    def from_tuple(cls, candinfo: Any) -> "CandidateInfo":
        try:
            size = len(candinfo)
        except TypeError as exc:
            raise ValueError("candinfo must be a tuple-like object") from exc

        if size < 4:
            raise ValueError(f"Unsupported candinfo length: {size}")

        dm, toa, freq_start, freq_end = candinfo[:4]
        dmt_idx = 0
        ref_toa = toa
        bbox = None
        confidence = None

        extras = list(candinfo[4:])
        if extras and isinstance(extras[0], (int, float)):
            dmt_idx = int(extras.pop(0))

        for extra in extras:
            if _is_bbox(extra):
                bbox = tuple(float(item) for item in extra)  # type: ignore[arg-type]
                continue
            if isinstance(extra, Mapping):
                value = extra.get("confidence")
                if value is not None:
                    confidence = float(value)
                continue
            if isinstance(extra, (int, float)):
                ref_toa = float(extra)

        return cls(dm, toa, freq_start, freq_end, dmt_idx, ref_toa, bbox, confidence)


@dataclass(frozen=True)
class DmPlotConfig:
    minpercentile: float = 0.1
    maxpercentile: float = 99.99


@dataclass(frozen=True)
class SpecPlotConfig:
    minpercentile: float = 0.1
    maxpercentile: float = 99.99
    tband: float = 300.0
    mode: str = "subband"
    subtsamp: int = 4
    subfreq: int = 128
    dtrend: bool = False
    norm: bool = True
    savetype: str = "png"
    snr_boxcar_max_ms: float | None = 30.0
    onlyspec: bool = False
    gc_collect_every_files: int = 10


def ensure_candidate_info(candinfo: Any) -> CandidateInfo:
    if isinstance(candinfo, CandidateInfo):
        return candinfo
    return CandidateInfo.from_tuple(candinfo)


def _is_bbox(value: Any) -> bool:
    try:
        if len(value) != 4:
            return False
    except TypeError:
        return False
    return not isinstance(value, (str, bytes, Mapping))


def ensure_dmt_config(dmtconfig: Any) -> DmPlotConfig:
    if isinstance(dmtconfig, DmPlotConfig):
        return dmtconfig
    if dmtconfig is None:
        return DmPlotConfig()
    if not isinstance(dmtconfig, Mapping):
        raise TypeError("dmtconfig must be a mapping or DmPlotConfig")
    return DmPlotConfig(**_filter_kwargs(DmPlotConfig, dmtconfig))


def ensure_spec_config(specconfig: Any) -> SpecPlotConfig:
    if isinstance(specconfig, SpecPlotConfig):
        return specconfig
    if specconfig is None:
        return SpecPlotConfig()
    if not isinstance(specconfig, Mapping):
        raise TypeError("specconfig must be a mapping or SpecPlotConfig")

    data = dict(specconfig)
    if "detrend" in data and "dtrend" not in data:
        data["dtrend"] = data["detrend"]
    if "boxcar_max_ms" in data and "snr_boxcar_max_ms" not in data:
        data["snr_boxcar_max_ms"] = data["boxcar_max_ms"]
    return SpecPlotConfig(**_filter_kwargs(SpecPlotConfig, data))


def _filter_kwargs(cls: type, mapping: Mapping[str, Any]) -> dict:
    field_names = {field.name for field in fields(cls)}
    return {key: value for key, value in mapping.items() if key in field_names}
