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

    @classmethod
    def from_tuple(cls, candinfo: Any) -> "CandidateInfo":
        try:
            size = len(candinfo)
        except TypeError as exc:
            raise ValueError("candinfo must be a tuple-like object") from exc

        if size == 7:
            dm, toa, freq_start, freq_end, dmt_idx, bbox, ref_toa = candinfo
            return cls(dm, toa, freq_start, freq_end, dmt_idx, ref_toa, bbox)
        if size == 6:
            dm, toa, freq_start, freq_end, dmt_idx, ref_toa = candinfo
            return cls(dm, toa, freq_start, freq_end, dmt_idx, ref_toa, None)
        if size == 5:
            dm, toa, freq_start, freq_end, dmt_idx = candinfo
            return cls(dm, toa, freq_start, freq_end, dmt_idx, toa, None)

        raise ValueError(f"Unsupported candinfo length: {size}")


@dataclass(frozen=True)
class DmPlotConfig:
    minpercentile: float = 5.0
    maxpercentile: float = 99.9


@dataclass(frozen=True)
class SpecPlotConfig:
    minpercentile: float = 5.0
    maxpercentile: float = 99.0
    tband: float = 50.0
    mode: str = "subband"
    subtsamp: int = 4
    subfreq: int = 128
    dtrend: bool = False
    norm: bool = True
    savetype: str = "png"
    snr_boxcar_max_ms: float | None = None
    onlyspec: bool = False


def ensure_candidate_info(candinfo: Any) -> CandidateInfo:
    if isinstance(candinfo, CandidateInfo):
        return candinfo
    return CandidateInfo.from_tuple(candinfo)


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
