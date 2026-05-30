"""
Multi-view DMT training image generation from real filterbank/psrfits data.

Generates training images and YOLO labels based on actual pulse injections.
Each pulse gets its own dedispersion, consistent with detector.mutidetect().

Key Design:
- For each pulse and strategy: build concrete dedisperse_spec() parameters
- Use resulting DmTime objects directly (similar to detector.mutidetect)
- Generate multiple views from strategy-specific DM/time parameters
- YOLO coordinates: x→time, y→DM (following detector convention)
"""

import os
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import cv2
from tqdm import tqdm

from ..io.data import SpectrumBase
from ..io.filterbank import Filterbank
from ..io.psrfits import PsrFits
from ..dedispered import dedisperse_spec
from ..dmtime import DmTime
from ..logger import logger

SUPPORTED_EXTENSIONS = {'.fil': Filterbank, '.fits': PsrFits}


class ViewStrategy(Enum):
    """
    View generation strategies with corresponding DM/time ranges.
    
    Each strategy produces images with different DM/time characteristics to train
    the detector model on various pulse representations.
    
    Attributes:
        OPERATIONAL: Use segment's full dm_range, standard time_span. Baseline view.
        OFF_CENTER: Adjusted dm_range (60% of segment span) and adjusted time_span 
                   to shift pulse position toward the center of the window.
        SCALE_NARROW: Narrower dm_range and shorter time_span (50% of baseline).
                     Tests narrow-band pulse detection.
        SCALE_WIDE: Wider dm_range and longer time_span (200% of baseline).
                   Tests wide-band pulse detection.
        BOUNDARY_DM: Cross adjacent DM segments. Tests boundary behavior between segments.
        BOUNDARY_TIME: Pulse positioned near time window boundary (5-15% or 85-95%).
                      Tests edge-case detection near temporal boundaries.
        BOUNDARY_DM_ASYMMETRIC: Asymmetric boundary crossing (requires pulse near boundary).
                               Tests asymmetric segment boundary crossing.
        JITTER: Small random perturbations (±20%) on DM range and time span.
               Tests robustness to parameter variations.
    """
    OPERATIONAL = "operational"
    OFF_CENTER = "off_center"
    SCALE_NARROW = "scale_narrow"
    SCALE_WIDE = "scale_wide"
    BOUNDARY_DM = "boundary_dm"
    BOUNDARY_TIME = "boundary_time"
    BOUNDARY_DM_ASYMMETRIC = "boundary_dm_asymmetric"
    JITTER = "jitter"


@dataclass
class StrategyConfig:
    """
    Configuration for a view strategy - defines DM range and time span.
    
    Used internally by MultiViewImageGenerator to store the parameterization
    of each ViewStrategy. Determines how DM ranges and time windows are 
    calculated for dedispersion.
    
    Attributes:
        strategy: The ViewStrategy this config applies to.
        dm_range_factor: Multiplicative factor for DM range span (1.0 = full segment).
        dm_center_offset: Fractional offset of DM range center relative to span.
                         Range: -0.5 to 0.5. Negative shifts left, positive shifts right.
        time_span_factor: Multiplicative factor for time window span.
        time_center_offset: Fractional offset of time center relative to span.
                           Similar to dm_center_offset but for time dimension.
        cross_boundary: If True, this strategy intentionally crosses DM segment boundaries.
    """
    strategy: ViewStrategy
    dm_range_factor: float = 1.0
    dm_center_offset: float = 0.0
    time_span_factor: float = 1.0
    time_center_offset: float = 0.0
    cross_boundary: bool = False


@dataclass
class DMSegmentConfig:
    """
    Configuration for a DM search segment.
    
    Represents a contiguous range of DM values used for dedispersion searches.
    Multiple segments together define the global DM search space.
    
    Attributes:
        name: Human-readable name for this segment (e.g., "low", "mid", "high").
        dm_low: Minimum DM value in this segment (pc/cm³).
        dm_high: Maximum DM value in this segment (pc/cm³).
        dm_step: Default DM step size for dedispersion in this segment.
    """
    name: str
    dm_low: float
    dm_high: float
    dm_step: float
    
    def contains(self, dm: float) -> bool:
        """
        Check if a DM value falls within this segment.
        
        Args:
            dm: DM value to check (pc/cm³).
            
        Returns:
            True if dm_low <= dm <= dm_high, False otherwise.
        """
        return self.dm_low <= dm <= self.dm_high


@dataclass
class ViewConfig:
    """
    Configuration for generating multiple views.
    
    Central configuration class that specifies all parameters needed to generate
    training images. Passed to MultiViewImageGenerator to control view generation.
    
    Attributes:
        dm_segments: List of DM segment configurations defining the search space.
        time_span: Default time window span in seconds for dedispersion (default: 1.0).
        img_height: Height of output images in pixels (default: 512).
        img_width: Width of output images in pixels (default: 512).
        time_downsample: Factor to downsample in time dimension during dedispersion
                        (default: 1, no downsampling).
        freq_start: Start frequency for frequency range in MHz (default: 1001.0).
        freq_end: End frequency for frequency range in MHz (default: 1500.0).
        scale_narrow_factor: Multiplicative factor for SCALE_NARROW strategy 
                            (default: 0.5, half size).
        scale_wide_factor: Multiplicative factor for SCALE_WIDE strategy 
                          (default: 2.0, double size).
        boundary_margin_dm: DM margin around segment boundaries for boundary detection
                           in pc/cm³ (default: 5.0).
        boundary_margin_time: Time margin around boundaries for time-based boundary
                             detection (default: 0.2).
        boundary_dm_grid: DM grid spacing for boundary crossing calculations
                         (default: 10.0, not currently used).
        boundary_dm_tolerance: DM tolerance for BOUNDARY_DM strategy in pc/cm³
                              (default: 20.0). Controls range overlap size.
        jitter_factor: Maximum random perturbation factor for JITTER strategy
                      (default: 0.2, ±20% variation).
        boundary_time_edge_fraction: Fraction of time window considered "edge" for
                                    BOUNDARY_TIME strategy (default: 0.1, 5-15%).
        enabled_strategies: Optional list of strategy names to generate. If None,
                           all strategies are generated. Names should match
                           ViewStrategy enum values (lowercase).
    """
    dm_segments: List[DMSegmentConfig]
    time_span: float = 1.0
    img_height: int = 512
    img_width: int = 512
    time_downsample: int = 1
    freq_start: float = 1001.0
    freq_end: float = 1500.0
    scale_narrow_factor: float = 0.5
    scale_wide_factor: float = 2.0
    boundary_margin_dm: float = 5.0
    boundary_margin_time: float = 0.2
    boundary_dm_grid: float = 10.0
    boundary_dm_tolerance: float = 20.0
    jitter_factor: float = 0.2
    boundary_time_edge_fraction: float = 0.1
    enabled_strategies: Optional[List[str]] = None


@dataclass
class PulseInjectionParams:
    """
    Parameters for an injected pulse.
    
    Describes the characteristics of a single injected or detected pulse that
    will be used to generate training views.
    
    Attributes:
        dm_inject: Dispersion measure of pulse in pc/cm³.
        toa_inject: Time of arrival of pulse (relative to spectrum start) in seconds.
        snr: Signal-to-noise ratio of the pulse.
        width: Pulse width in seconds or milliseconds (depending on convention).
    """
    dm_inject: float
    toa_inject: float
    snr: float
    width: float


@dataclass
class DedisperseParams:
    """
    Concrete parameters for one dedisperse_spec() call.
    
    Represents the fully-resolved parameters passed to dedisperse_spec() for
    a specific view generation. Built by _build_dedisperse_params() from
    strategy configuration and pulse properties.
    
    Attributes:
        dm_low: Lower DM boundary for dedispersion in pc/cm³.
        dm_high: Upper DM boundary for dedispersion in pc/cm³.
        dm_step: DM step size for dedispersion trial (pc/cm³).
        time_downsample: Time downsampling factor (1 = no downsampling).
        t_sample: Time sample duration in seconds (chunk size for chunked dedispersion).
    """
    dm_low: float
    dm_high: float
    dm_step: float
    time_downsample: int
    t_sample: float


@dataclass
class ViewMetadata:
    """
    Metadata for a generated view.
    
    Contains all information about a generated view image, including pulse
    parameters, DM-time (DMT) coordinates, bounding box information, and
    YOLO label coordinates. Used for training dataset annotation.
    
    Attributes:
        view_id: Unique identifier for this view (format: pulse_id_strategy_index).
        view_strategy: Name of the ViewStrategy used (e.g., "operational").
        dm_inject: Injected DM value in pc/cm³.
        toa_inject: Injected time of arrival in seconds.
        snr: Signal-to-noise ratio.
        width: Pulse width.
        bbox_x: YOLO center x-coordinate (normalized, 0-1), time dimension.
        bbox_y: YOLO center y-coordinate (normalized, 0-1), DM dimension.
        bbox_w: YOLO bbox width (normalized, 0-1). Fixed at 0.2.
        bbox_h: YOLO bbox height (normalized, 0-1). Fixed at 0.2.
        dm_segment_name: Name of the DM segment containing the pulse.
        dmt_dm_low: DM lower boundary of the DmTime object used.
        dmt_dm_high: DM upper boundary of the DmTime object used.
        dmt_tstart: Time start of the DmTime object in seconds.
        dmt_tend: Time end of the DmTime object in seconds.
        dm_num_experiments: Number of DM experiments (trials) in the DMT.
        img_height: Height of generated image in pixels (default: 512).
        img_width: Width of generated image in pixels (default: 512).
        bbox_x_raw: Initial bbox_x before time refinement (for debugging).
        bbox_x_refined: Refined bbox_x from time profile analysis, if applied.
        bbox_refine_used: True if time refinement was applied successfully.
    """
    view_id: str
    view_strategy: str
    dm_inject: float
    toa_inject: float
    snr: float
    width: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    dm_segment_name: str
    dmt_dm_low: float
    dmt_dm_high: float
    dmt_tstart: float
    dmt_tend: float
    dm_num_experiments: int
    img_height: int = 512
    img_width: int = 512
    bbox_x_raw: float = 0.0
    bbox_x_refined: Optional[float] = None
    bbox_refine_used: bool = False


class MultiViewImageGenerator:
    """
    Generate multiple views of DMT images for training.
    
    This is the main class for generating training images from radio pulsar data.
    It takes a ViewConfig specifying DM segments and view parameters, and generates
    multiple dedispersed views for each pulse using different strategies.
    
    Workflow:
        1. Initialize with ViewConfig specifying segments and parameters
        2. For each pulse: call generate_all_views() or generate_training_set()
        3. For each strategy: build dedisperse_spec() parameters
        4. Call dedisperse_spec() to get DmTime objects
        5. Extract and normalize images from DmTime data
        6. Save images with corresponding labels
    
    Key Design Decisions:
        - Each strategy maps to specific DM/time parameter configurations
        - JITTER strategy uses random perturbations (executed at runtime)
        - BOUNDARY strategies require pulses near segment boundaries
        - YOLO coordinates: x=time (0-1 normalized), y=DM (0-1 normalized)
        - Image normalization: per-image min-max normalization, NaN/Inf handling
        - Bbox refinement: optional time-coordinate refinement from DM-band profile
    """
    
    def __init__(self, view_config: ViewConfig):
        """
        Initialize the generator with configuration.
        
        Args:
            view_config: ViewConfig object specifying DM segments, image parameters,
                        and strategy-specific settings.
                        
        Raises:
            ValueError: If view_config has no DM segments.
        """
        self.config = view_config
        self._validate_config()
        self.strategy_configs = self._create_strategy_configs()
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Checks that essential configuration parameters are valid.
        Currently validates that at least one DM segment is defined.
        
        Raises:
            ValueError: If configuration is invalid (e.g., no DM segments).
        """
        if not self.config.dm_segments:
            raise ValueError("At least one DM segment must be configured")
    
    def _create_strategy_configs(self) -> Dict[ViewStrategy, StrategyConfig]:
        """
        Create DM/time range configurations for each strategy.
        
        Initializes the strategy configuration dictionary with pre-computed parameters
        for each ViewStrategy. These define how DM ranges and time windows are scaled
        and offset for each strategy.
        
        Returns:
            Dictionary mapping ViewStrategy enum members to StrategyConfig objects.
            Each config specifies DM/time scaling factors and offsets.
        """
        return {
            ViewStrategy.OPERATIONAL: StrategyConfig(
                strategy=ViewStrategy.OPERATIONAL,
                dm_range_factor=1.0,
                dm_center_offset=0.0,
                time_span_factor=1.0,
                time_center_offset=0.0,
            ),
            ViewStrategy.OFF_CENTER: StrategyConfig(
                strategy=ViewStrategy.OFF_CENTER,
                dm_range_factor=0.6,
                dm_center_offset=0.3,
                time_span_factor=1.0,
                time_center_offset=0.25,
            ),
            ViewStrategy.SCALE_NARROW: StrategyConfig(
                strategy=ViewStrategy.SCALE_NARROW,
                dm_range_factor=self.config.scale_narrow_factor,
                dm_center_offset=0.0,
                time_span_factor=self.config.scale_narrow_factor,
                time_center_offset=0.0,
            ),
            ViewStrategy.SCALE_WIDE: StrategyConfig(
                strategy=ViewStrategy.SCALE_WIDE,
                dm_range_factor=self.config.scale_wide_factor,
                dm_center_offset=0.0,
                time_span_factor=self.config.scale_wide_factor,
                time_center_offset=0.0,
            ),
            ViewStrategy.BOUNDARY_DM: StrategyConfig(
                strategy=ViewStrategy.BOUNDARY_DM,
                dm_range_factor=1.5,
                dm_center_offset=0.0,
                time_span_factor=1.0,
                time_center_offset=0.0,
                cross_boundary=True,
            ),
            ViewStrategy.BOUNDARY_TIME: StrategyConfig(
                strategy=ViewStrategy.BOUNDARY_TIME,
                dm_range_factor=1.0,
                dm_center_offset=0.0,
                time_span_factor=1.0,
                time_center_offset=0.35,
            ),
            ViewStrategy.BOUNDARY_DM_ASYMMETRIC: StrategyConfig(
                strategy=ViewStrategy.BOUNDARY_DM_ASYMMETRIC,
                dm_range_factor=1.2,
                dm_center_offset=0.0,
                time_span_factor=1.0,
                time_center_offset=0.0,
                cross_boundary=True,
            ),
            ViewStrategy.JITTER: StrategyConfig(
                strategy=ViewStrategy.JITTER,
                dm_range_factor=1.0,
                dm_center_offset=0.0,
                time_span_factor=1.0,
                time_center_offset=0.0,
            ),
        }
    
    def _find_segment_for_dm(self, dm: float) -> Optional[DMSegmentConfig]:
        """
        Find DM segment containing the given DM value.
        
        Iterates through configured DM segments to find the one that contains
        the specified DM value.
        
        Args:
            dm: DM value in pc/cm³ to locate.
            
        Returns:
            DMSegmentConfig containing the DM, or None if DM is outside all segments.
        """
        for segment in self.config.dm_segments:
            if segment.contains(dm):
                return segment
        return None
    
    def _global_dm_bounds(self) -> Tuple[float, float]:
        """
        Return the global DM bounds covering all configured segments.
        
        Returns:
            Tuple of (dm_low, dm_high) representing the full DM range.
        """
        return self.config.dm_segments[0].dm_low, self.config.dm_segments[-1].dm_high

    def _clip_dm_range(self, dm_low: float, dm_high: float) -> Tuple[float, float]:
        """
        Clip a DM range to the configured global search bounds.
        
        Ensures that any generated DM range stays within the global bounds
        defined by the first and last DM segments.
        
        Args:
            dm_low: Proposed lower DM boundary.
            dm_high: Proposed upper DM boundary.
            
        Returns:
            Tuple of (clipped_dm_low, clipped_dm_high) within global bounds.
        """
        global_low, global_high = self._global_dm_bounds()
        return max(dm_low, global_low), min(dm_high, global_high)

    def _range_around_dm(
        self,
        dm: float,
        span: float,
        center_offset: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Build a DM range around the injected pulse and keep the pulse inside it.
        
        Constructs a symmetric DM range of specified span, optionally offset from
        the pulse DM. Ensures the pulse DM always falls within the returned range
        by shifting if necessary. Range is clipped to global bounds.
        
        Args:
            dm: Center DM value (pulse location) in pc/cm³.
            span: Width of DM range to generate in pc/cm³.
            center_offset: Fractional offset of range center from dm. 
                          0.0 = centered on dm, ±0.5 = extreme offsets.
                          
        Returns:
            Tuple of (dm_low, dm_high) with dm guaranteed in range.
        """
        center = dm + center_offset * span
        dm_low = center - span / 2
        dm_high = center + span / 2
        dm_low, dm_high = self._clip_dm_range(dm_low, dm_high)

        if dm < dm_low:
            shift = dm_low - dm
            dm_low, dm_high = self._clip_dm_range(dm_low - shift, dm_high - shift)
        elif dm > dm_high:
            shift = dm - dm_high
            dm_low, dm_high = self._clip_dm_range(dm_low + shift, dm_high + shift)

        return dm_low, dm_high

    def _calculate_dm_range_for_strategy(
        self,
        segment: DMSegmentConfig,
        dm_inject: float,
        strategy_cfg: StrategyConfig
    ) -> Tuple[float, float]:
        """
        Calculate DM range for a strategy.
        
        Uses the strategy configuration to scale and offset the segment's base
        DM range, while ensuring the pulse DM remains within bounds.
        
        Args:
            segment: DM segment containing the pulse.
            dm_inject: Injected pulse DM in pc/cm³.
            strategy_cfg: StrategyConfig with scaling and offset parameters.
        
        Returns:
            Tuple of (dm_low, dm_high) for this strategy's dedispersion.
        """
        base_span = segment.dm_high - segment.dm_low
        new_span = base_span * strategy_cfg.dm_range_factor
        return self._range_around_dm(
            dm_inject,
            new_span,
            center_offset=strategy_cfg.dm_center_offset,
        )
    
    def _is_near_dm_boundary(self, dm: float) -> bool:
        """
        Check if DM is near a segment boundary.
        
        Used to determine if a pulse is eligible for BOUNDARY_* strategies.
        
        Args:
            dm: DM value to check in pc/cm³.
            
        Returns:
            True if dm is within boundary_margin_dm of any segment edge.
        """
        margin = self.config.boundary_margin_dm
        for segment in self.config.dm_segments:
            if abs(dm - segment.dm_low) < margin or abs(dm - segment.dm_high) < margin:
                return True
        return False
    
    def _find_nearest_dm_boundary(
        self,
        dm: float,
        segment: DMSegmentConfig
    ) -> Optional[Tuple[float, str]]:
        """
        Find the nearest segment boundary that has an adjacent segment.
        
        Searches for DM segment boundaries that are adjacent to the current segment
        and returns the nearest one to the pulse DM. Used by BOUNDARY_* strategies.
        
        Args:
            dm: Pulse DM in pc/cm³.
            segment: Current DM segment.
        
        Returns:
            Tuple of (boundary_dm, side) where:
            - boundary_dm: DM value of the boundary
            - side: "lower" for segment.dm_low, "upper" for segment.dm_high
            Returns None if no adjacent segments exist.
        """
        candidates = []
        for other_seg in self.config.dm_segments:
            if other_seg is segment:
                continue
            if other_seg.dm_low < segment.dm_low <= other_seg.dm_high or other_seg.dm_high <= segment.dm_low:
                candidates.append((abs(dm - segment.dm_low), segment.dm_low, "lower"))
                break
        for other_seg in self.config.dm_segments:
            if other_seg is segment:
                continue
            if other_seg.dm_low <= segment.dm_high < other_seg.dm_high or other_seg.dm_low >= segment.dm_high:
                candidates.append((abs(dm - segment.dm_high), segment.dm_high, "upper"))
                break
        if not candidates:
            return None
        _, boundary, side = min(candidates, key=lambda item: item[0])
        return boundary, side

    def _calculate_boundary_dm_range(
        self,
        segment: DMSegmentConfig,
        dm_inject: float,
        span_factor: float,
        asymmetric: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """
        Build a DM range that contains the pulse and crosses an adjacent segment boundary.
        
        Used by BOUNDARY_DM_ASYMMETRIC strategy to create views where the DM search
        range straddles a segment boundary. Ensures the pulse remains visible while
        crossing into adjacent DM territory.
        
        Args:
            segment: Current DM segment.
            dm_inject: Pulse DM in pc/cm³.
            span_factor: Factor to scale base segment span.
            asymmetric: If False, center range on mid-point between pulse and boundary.
                       If True, asymmetrically place range to ensure pulse visibility.
        
        Returns:
            Tuple of (dm_low, dm_high) crossing the boundary, or None if:
            - No adjacent segment exists
            - Resulting range doesn't contain pulse and cross boundary
        """
        nearest = self._find_nearest_dm_boundary(dm_inject, segment)
        if nearest is None:
            return None

        boundary, side = nearest
        base_span = segment.dm_high - segment.dm_low
        span = base_span * span_factor

        if asymmetric:
            margin = self.config.boundary_margin_dm
            if side == "lower":
                dm_low = min(boundary - margin, dm_inject - margin)
                dm_high = dm_low + span
                if dm_high < dm_inject:
                    dm_high = dm_inject + margin
            else:
                dm_high = max(boundary + margin, dm_inject + margin)
                dm_low = dm_high - span
                if dm_low > dm_inject:
                    dm_low = dm_inject - margin
        else:
            center = (dm_inject + boundary) / 2
            dm_low = center - span / 2
            dm_high = center + span / 2

        dm_low, dm_high = self._clip_dm_range(dm_low, dm_high)
        if not (dm_low <= dm_inject <= dm_high and dm_low < boundary < dm_high):
            return None
        return dm_low, dm_high

    def _time_span_with_pulse_fraction(
        self,
        toa_inject: float,
        preferred_span: float,
        target_fraction: float
    ) -> float:
        """
        Choose t_sample so dedisperse_spec's sliding chunks place TOA near a target fraction.

        dedisperse_spec creates chunks [idx * 0.9 * t_sample, idx * 0.9 * t_sample + t_sample].
        This method adjusts t_sample to position the pulse near a target fraction
        within its chunk (e.g., 0.25 = 25% into the chunk, 0.75 = 75% into).
        
        Args:
            toa_inject: Pulse TOA in seconds.
            preferred_span: Preferred time window span in seconds.
            target_fraction: Target position within chunk [0.05, 0.95].
                           0.5 = centered, 0.1 = near start, 0.9 = near end.
        
        Returns:
            t_sample value adjusted to position pulse at target_fraction.
            Returns preferred_span if no suitable t_sample found.
        
        Algorithm:
            For chunk idx containing toa_inject with target_fraction f:
                toa_inject = (idx * 0.9 + f) * t_sample
                t_sample = toa_inject / (idx * 0.9 + f)
            Searches idx ± 10 to find closest match to preferred_span.
        """
        if toa_inject <= 0 or preferred_span <= 0:
            return preferred_span

        target_fraction = float(np.clip(target_fraction, 0.05, 0.95))
        approx_idx = max(int(toa_inject / (0.9 * preferred_span)), 0)
        candidates = []
        for idx in range(max(0, approx_idx - 10), approx_idx + 11):
            denom = idx * 0.9 + target_fraction
            if denom <= 0:
                continue
            t_sample = toa_inject / denom
            if t_sample > 0:
                candidates.append(t_sample)

        if not candidates:
            return preferred_span
        return min(candidates, key=lambda value: abs(value - preferred_span))

    def _time_spans_for_strategy(
        self,
        pulse_params: PulseInjectionParams,
        strategy_cfg: StrategyConfig,
    ) -> List[float]:
        """
        Calculate t_sample values for strategy-specific dedisperse_spec() calls.
        
        Returns a list of time span values appropriate for a given strategy.
        Most strategies return a single value, but BOUNDARY_TIME returns a
        randomly selected edge position.
        
        Args:
            pulse_params: Pulse parameters including toa_inject.
            strategy_cfg: Configuration for the current strategy.
        
        Returns:
            List of t_sample values to try for this strategy.
        """
        preferred_span = self.config.time_span * strategy_cfg.time_span_factor
        strategy = strategy_cfg.strategy

        if strategy == ViewStrategy.OFF_CENTER:
            target_fraction = 0.25 if strategy_cfg.time_center_offset >= 0 else 0.75
            return [self._time_span_with_pulse_fraction(
                pulse_params.toa_inject,
                preferred_span,
                target_fraction,
            )]

        if strategy == ViewStrategy.BOUNDARY_TIME:
            # Random position at 5-15% or 85-95% of time window
            edge_fraction = self.config.boundary_time_edge_fraction  # 0.1 means 5-15%
            choice = np.random.choice([0, 1])
            if choice == 0:
                # Low edge: 5-15%
                target_fraction = np.random.uniform(edge_fraction / 2, edge_fraction)
            else:
                # High edge: 85-95%
                target_fraction = np.random.uniform(1.0 - edge_fraction, 1.0 - edge_fraction / 2)
            
            return [self._time_span_with_pulse_fraction(
                pulse_params.toa_inject,
                preferred_span,
                target_fraction,
            )]

        return [preferred_span]

    def _boundary_dm_ranges(self, dm_inject: float, segment: DMSegmentConfig) -> List[Tuple[float, float]]:
        """
        Build DM ranges with tolerance for BOUNDARY_DM strategy.
        
        Generates two overlapping DM ranges that bracket the pulse DM. This creates
        views where the pulse appears at different positions within the DM axis,
        testing detector robustness to boundary conditions.
        
        Example with DM=740, tolerance=20, span=600:
            - Range 1: (720, 1340) - narrow band containing DM, wider span
            - Range 2: (140, 760)  - wide band spanning to boundary, narrow span
        
        Args:
            dm_inject: Pulse DM in pc/cm³.
            segment: DM segment containing the pulse (unused, for context).
        
        Returns:
            List of (dm_low, dm_high) tuples that satisfy:
            1. Contain the pulse DM (dm_low <= dm_inject <= dm_high)
            2. Stay within global bounds
            3. Have non-zero span
        """
        tolerance = self.config.boundary_dm_tolerance
        span = 600.0  # Fixed DM span for boundary ranges
        global_low, global_high = self._global_dm_bounds()
        
        # Range 1: from (dm - tolerance) to (dm + span)
        dm_low_1 = max(dm_inject - tolerance, global_low)
        dm_high_1 = min(dm_inject + span, global_high)
        
        # Range 2: from (dm - span) to (dm + tolerance)
        dm_low_2 = max(dm_inject - span, global_low)
        dm_high_2 = min(dm_inject + tolerance, global_high)
        
        ranges = [(dm_low_1, dm_high_1), (dm_low_2, dm_high_2)]
        
        return [
            (dm_low, dm_high)
            for dm_low, dm_high in ranges
            if dm_low <= dm_inject <= dm_high and dm_high > dm_low
        ]

    def _build_dedisperse_params(
        self,
        pulse_params: PulseInjectionParams,
        segment: DMSegmentConfig,
        strategy: ViewStrategy,
    ) -> List[DedisperseParams]:
        """
        Translate a view strategy into concrete dedisperse_spec() parameters.
        
        For a given pulse and strategy, builds one or more DedisperseParams objects
        that will be passed to dedisperse_spec() to generate DM-time chunks.
        Different strategies generate different DM ranges and time spans.
        
        Args:
            pulse_params: Pulse parameters including dm_inject and toa_inject.
            segment: DMSegmentConfig containing the pulse.
            strategy: ViewStrategy to apply.
        
        Returns:
            List of DedisperseParams. Can be empty if strategy doesn't apply
            (e.g., BOUNDARY_DM_ASYMMETRIC for pulse not near boundary).
        
        Strategy Behavior:
            - OPERATIONAL: Full segment DM range, standard time span
            - OFF_CENTER: 60% DM range centered offset, standard time span
            - SCALE_NARROW: 50% DM and time ranges, centered on pulse
            - SCALE_WIDE: 200% DM and time ranges, centered on pulse
            - BOUNDARY_DM: Two DM ranges that bracket the pulse DM
            - BOUNDARY_TIME: Standard DM range, time positioned at edge (5-15% or 85-95%)
            - BOUNDARY_DM_ASYMMETRIC: Asymmetric DM range crossing segment boundary
            - JITTER: Random ±20% perturbations on DM and time ranges
        """
        strategy_cfg = self.strategy_configs[strategy]

        if strategy == ViewStrategy.JITTER:
            # Use ±20% random jitter
            jitter_factor = self.config.jitter_factor
            strategy_cfg = StrategyConfig(
                strategy=ViewStrategy.JITTER,
                dm_range_factor=np.random.uniform(1.0 - jitter_factor, 1.0 + jitter_factor),
                time_span_factor=np.random.uniform(1.0 - jitter_factor, 1.0 + jitter_factor),
            )

        if strategy in [ViewStrategy.OPERATIONAL, ViewStrategy.BOUNDARY_TIME]:
            dm_ranges = [(segment.dm_low, segment.dm_high)]
        elif strategy == ViewStrategy.BOUNDARY_DM:
            dm_ranges = self._boundary_dm_ranges(pulse_params.dm_inject, segment)
        elif strategy == ViewStrategy.BOUNDARY_DM_ASYMMETRIC:
            if not self._is_near_dm_boundary(pulse_params.dm_inject):
                return []
            dm_range = self._calculate_boundary_dm_range(
                segment,
                pulse_params.dm_inject,
                strategy_cfg.dm_range_factor,
                asymmetric=True,
            )
            if dm_range is None:
                return []
            dm_ranges = [dm_range]
        else:
            dm_ranges = [
                self._calculate_dm_range_for_strategy(
                    segment,
                    pulse_params.dm_inject,
                    strategy_cfg,
                )
            ]

        time_spans = self._time_spans_for_strategy(
            pulse_params,
            strategy_cfg,
        )
        params = []
        for dm_low, dm_high in dm_ranges:
            if dm_high <= dm_low:
                continue
            dm_span = dm_high - dm_low
            dm_step = dm_span / 600
            for strategy_time_span in time_spans:
                if strategy_time_span <= 0:
                    continue
                params.append(DedisperseParams(
                    dm_low=dm_low,
                    dm_high=dm_high,
                    dm_step=dm_step,
                    time_downsample=self.config.time_downsample,
                    t_sample=strategy_time_span,
                ))
        return params
    
    def _calculate_bbox_from_dmt(
        self,
        dm_inject: float,
        toa_inject: float,
        dmt: DmTime,
        pulse_width: float,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate YOLO bounding box based on DmTime coordinates.
        
        Converts pulse parameters to normalized YOLO box coordinates using the same
        convention as detector.mutidetect(). Optionally refines the time coordinate
        by analyzing the image's time profile.
        
        Args:
            dm_inject: Injected DM value in pc/cm³.
            toa_inject: Injected TOA in seconds (detector convention: referenced to
                       highest frequency arrival).
            dmt: DmTime object containing the dedispersed data.
            pulse_width: Pulse width (unused in current implementation).
            image: Optional 2D image array for time refinement. If provided and valid,
                  the time coordinate may be refined by finding the peak in the
                  smoothed time profile.
        
        Returns:
            Dictionary with keys:
                - bbox_x: Normalized center x-coordinate [0, 1] (time dimension)
                - bbox_y: Normalized center y-coordinate [0, 1] (DM dimension)
                - bbox_w: Width of bbox in normalized coordinates (fixed at 0.2)
                - bbox_h: Height of bbox in normalized coordinates (fixed at 0.2)
                - bbox_x_raw: Initial bbox_x before refinement (for debugging)
                - bbox_x_refined: Refined bbox_x if refinement applied, else None
                - bbox_refine_used: True if time refinement was successfully applied
        
        Coordinate Convention:
            - x (time) = (toa - tstart) / (tend - tstart)
            - y (DM) = (dm - dm_low) / (dm_high - dm_low)
        """
        # CSV TOA is referenced to the highest-frequency arrival time and is
        # the detector coordinate convention. Use it as the initial anchor.
        dm_rel = (dm_inject - dmt.dm_low) / (dmt.dm_high - dmt.dm_low)
        time_rel = (toa_inject - dmt.tstart) / (dmt.tend - dmt.tstart)
        
        # Clip to valid range
        dm_rel = np.clip(dm_rel, 0.0, 1.0)
        time_rel = np.clip(time_rel, 0.0, 1.0)
        raw_time_rel = float(time_rel)
        refined_time = None

        # Use the existing dataset convention: a fixed YOLO box size around the
        # signal center. Refine only the time coordinate from a smoothed
        # DM-band time profile; keep DM anchored to the CSV value.
        bbox_w = 0.2
        bbox_h = 0.2

        if image is not None and image.ndim == 2 and image.size > 0:
            refined_time = self._refine_time_from_profile(
                image,
                time_rel,
                dm_rel,
                bbox_h,
            )
            if refined_time is not None:
                time_rel = refined_time

        time_rel = np.clip(time_rel, bbox_w / 2, 1.0 - bbox_w / 2)
        dm_rel = np.clip(dm_rel, bbox_h / 2, 1.0 - bbox_h / 2)
        
        return {
            "bbox_x": time_rel,
            "bbox_y": dm_rel,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_x_raw": raw_time_rel,
            "bbox_x_refined": refined_time,
            "bbox_refine_used": refined_time is not None,
        }

    def _refine_time_from_profile(
        self,
        image: np.ndarray,
        time_rel: float,
        dm_rel: float,
        bbox_h: float,
    ) -> Optional[float]:
        """
        Refine time coordinate by finding the peak in the smoothed time profile.
        
        Extracts a horizontal band around the DM coordinate, averages across DM,
        applies median subtraction and smoothing, then finds the peak position.
        Returns normalized x-coordinate [0, 1] of the peak.
        
        Args:
            image: 2D image array (dm_axis × time_axis).
            time_rel: Initial time coordinate estimate [0, 1].
            dm_rel: DM coordinate [0, 1] around which to extract band.
            bbox_h: Height of band around dm_rel (used to determine band width).
        
        Returns:
            Refined time coordinate [0, 1] if successful, None if image is too sparse
            or contains no useful signal.
            
        Algorithm:
            1. Extract horizontal band ±5 pixels around dm_rel
            2. Average across DM to form 1D time profile
            3. Subtract median to center on signal
            4. Smooth with kernel (4% of image width)
            5. Find argmax position
            6. Return normalized coordinate
        """
        img = image.astype(np.float32)
        img_h, img_w = img.shape
        expected_y = int(round(dm_rel * (img_h - 1)))

        y_radius = 5
        y0 = max(0, expected_y - y_radius)
        y1 = min(img_h, expected_y + y_radius + 1)
        band = img[y0:y1, :]
        if band.size == 0 or not np.isfinite(band).any():
            return None
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

        profile = np.nanmean(band, axis=0)
        finite = profile[np.isfinite(profile)]
        if finite.size == 0:
            return None
        median = float(np.median(finite))

        profile = profile - median
        profile[profile < 0] = 0
        kernel_size = max(5, int(round(img_w * 0.04)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smooth = np.convolve(profile, kernel, mode="same")

        if not np.isfinite(smooth).any() or float(np.max(smooth)) <= 0:
            return None

        peak_x = int(np.argmax(smooth))
        return (peak_x + 0.5) / img_w
    
    def _find_best_dmt_for_pulse(
        self,
        dmtimes: List[DmTime],
        pulse_params: PulseInjectionParams
    ) -> Optional[DmTime]:
        """
        Find the best DmTime object that contains the pulse.
        
        Searches the list of DmTime objects for one that spans both the pulse's
        DM and TOA coordinates. Returns the first match found.
        
        Args:
            dmtimes: List of DmTime objects from a dedisperse_spec() call.
            pulse_params: PulseInjectionParams with dm_inject and toa_inject.
        
        Returns:
            DmTime object containing the pulse, or None if no match found.
        """
        for dmt in dmtimes:
            if (dmt.dm_low <= pulse_params.dm_inject <= dmt.dm_high and
                dmt.tstart <= pulse_params.toa_inject <= dmt.tend):
                return dmt
        return None
    
    def generate_view(
        self,
        dmtimes: List[DmTime],
        pulse_params: PulseInjectionParams,
        pulse_id: str,
        view_idx: int,
        strategy: ViewStrategy,
        dm_step: float
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        Generate a single view (image) for a pulse.
        
        Extracts the DmTime object containing the pulse, normalizes its data,
        resizes to target dimensions, and optionally refines the time coordinate.
        
        Args:
            dmtimes: List of DmTime objects from dedisperse_spec().
            pulse_params: Pulse parameters including dm_inject and toa_inject.
            pulse_id: Unique pulse identifier.
            view_idx: Sequential view index for this pulse.
            strategy: ViewStrategy used to generate this view.
            dm_step: DM step size (unused in current implementation).
        
        Returns:
            Tuple of (view_id, image_array) or None if pulse not found in DMT data.
            view_id format: "{pulse_id}_{strategy.value}_{view_idx:02d}"
            image_array: 2D numpy array with shape (img_height, img_width),
                        values in [0, 1], NaN/Inf handled.
        
        Processing Steps:
            1. Find DmTime containing the pulse
            2. Normalize image: min-max normalization across finite values
            3. Handle NaN/Inf: replace with 0
            4. Resize to target image dimensions using bilinear interpolation
            5. Optionally refine time coordinate from DM-band profile
        """
        segment = self._find_segment_for_dm(pulse_params.dm_inject)
        if segment is None:
            logger.warning(f"Pulse DM {pulse_params.dm_inject} not in any segment")
            return None
        
        # Find best DmTime for this pulse
        best_dmt = self._find_best_dmt_for_pulse(dmtimes, pulse_params)
        if best_dmt is None:
            logger.debug(f"No suitable DMT found for pulse {pulse_id}")
            return None
        
        dmt = best_dmt
        
        # Extract and normalize image from selected DmTime. DMT arrays can
        # contain NaN/Inf values; handle those before bbox refinement so the
        # label logic sees the same finite image that is eventually saved.
        img = dmt.data.astype(np.float32)
        finite = np.isfinite(img)
        if finite.any():
            finite_values = img[finite]
            img_min = float(finite_values.min())
            img_max = float(finite_values.max())
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img = np.clip(img, 0.0, 1.0)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        
        # Resize to target dimensions
        img_resized = cv2.resize(img, (self.config.img_width, self.config.img_height),
                                interpolation=cv2.INTER_LINEAR)
        
        # Return view_id and image only
        view_id = f"{pulse_id}_{strategy.value}_{view_idx:02d}"
        return view_id, img_resized
    
    def generate_all_views(
        self,
        spectrum,
        pulse_params: PulseInjectionParams,
        pulse_id: str,
        segment: DMSegmentConfig,
        freq_start: float,
        freq_end: float
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Generate all views for a single pulse using all enabled strategies.
        
        For each enabled ViewStrategy:
            1. Build concrete dedisperse_spec() parameters using _build_dedisperse_params()
            2. Call dedisperse_spec() with those parameters to get DmTime objects
            3. Find the DmTime chunk containing the pulse using _find_best_dmt_for_pulse()
            4. Generate image from that DmTime using generate_view()
            5. Clean up memory from the DmTime objects
        
        Args:
            spectrum: Spectrum object (Filterbank or PsrFits) with data and header.
            pulse_params: Pulse parameters for this injection.
            pulse_id: Unique pulse identifier.
            segment: DMSegmentConfig containing the pulse.
            freq_start: Start frequency for dedispersion in MHz.
            freq_end: End frequency for dedispersion in MHz.
        
        Returns:
            List of (view_id, image) tuples. Empty list if all strategies fail.
            Images are normalized to [0, 1] and resized to config dimensions.
        
        Strategy Filtering:
            If config.enabled_strategies is set, only those strategies are processed.
            Otherwise, all ViewStrategy enum members are processed.
        
        Error Handling:
            Errors in individual strategies are logged and skipped; the function
            continues with remaining strategies.
        """
        views = []
        strategies = list(ViewStrategy)
        
        # Filter strategies based on config
        if self.config.enabled_strategies is not None:
            enabled_names = set(s.lower() for s in self.config.enabled_strategies)
            strategies = [s for s in strategies if s.value.lower() in enabled_names]
        
        view_idx = 0

        for strategy in strategies:
            try:
                params_list = self._build_dedisperse_params(
                    pulse_params,
                    segment,
                    strategy,
                )
                if not params_list:
                    logger.debug(f"Skipping {strategy} view for pulse {pulse_id}")
                    continue

                for params in params_list:
                    strategy_dmtimes = dedisperse_spec(
                        spectrum,
                        params.dm_low, params.dm_high,
                        freq_start, freq_end,
                        params.dm_step,
                        time_downsample=params.time_downsample,
                        t_sample=params.t_sample
                    )
                    
                    if not strategy_dmtimes:
                        logger.warning(f"No DMT data for strategy {strategy} (pulse {pulse_id})")
                        continue
                    
                    # Generate view from this specific DMT
                    result = self.generate_view(
                        strategy_dmtimes, 
                        pulse_params, 
                        pulse_id, 
                        view_idx, 
                        strategy,
                        params.dm_step
                    )
                    
                    if result is not None:
                        views.append(result)
                        view_idx += 1
                    
                    # Clean up memory
                    del strategy_dmtimes
                
            except Exception as e:
                logger.error(f"Error generating {strategy} view for pulse {pulse_id}: {e}")
                continue
        
        return views
    
    def save_view(
        self,
        view_id: str,
        image: np.ndarray,
        output_dir: str
    ) -> str:
        """
        Save a generated view image to disk.
        
        Creates an "images" subdirectory in output_dir if it doesn't exist,
        converts the image to 8-bit unsigned integer, and saves as PNG.
        
        Args:
            view_id: Unique view identifier used as filename base.
            image: Image array with shape (height, width), values in [0, 1] or [0, 255].
            output_dir: Directory where "images" subdirectory will be created.
        
        Returns:
            Full file path of the saved image.
        
        File Convention:
            Saved as: {output_dir}/images/{view_id}.png
        """
        img_dir = Path(output_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        # Image filename
        img_name = f"{view_id}.png"
        img_path = img_dir / img_name
        
        # Save image (convert to 8-bit)
        if image.dtype != np.uint8:
            img_8bit = (image * 255).astype(np.uint8)
        else:
            img_8bit = image
        
        cv2.imwrite(str(img_path), img_8bit)
        return str(img_path)


def generate_training_set(
    spectrum_file: str,
    pulses: List[Tuple[float, float, float, float]],
    view_config: ViewConfig,
    output_dir: str,
    pulse_ids: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Generate full training set for multiple pulses (module-level API).
    
    High-level function that loads a spectrum file once, then generates
    multiple views for each input pulse. Provides statistics on the
    generation process.
    
    Args:
        spectrum_file: Path to filterbank (.fil) or PSRFITS (.fits) file.
        pulses: List of pulse specifications, each a tuple:
               (dm_inject, toa_inject, snr, width)
        view_config: ViewConfig specifying DM segments, view parameters, strategies.
        output_dir: Output directory where "images" subdirectory will be created.
        pulse_ids: Optional list of custom pulse IDs. If None, uses pulse_XXXX format.
    
    Returns:
        Dictionary with statistics:
            - total_pulses: Number of input pulses
            - total_views: Total views generated (across all pulses and strategies)
            - successful_views: Views successfully saved
            - failed_views: Views that failed to save
    
    Workflow:
        1. Load spectrum file (auto-detects format from extension)
        2. Create MultiViewImageGenerator with view_config
        3. For each pulse:
           a. Find DM segment containing pulse
           b. Call generate_all_views() to get all strategy-specific views
           c. Save each view using save_view()
           d. Update statistics
        4. Return aggregated statistics
    
    Supported Formats:
        - .fil: Filterbank format (Filterbank class)
        - .fits: PSRFITS format (PsrFits class)
    
    Logging:
        INFO: Spectrum loaded, spectrum characteristics, completion stats
        DEBUG: Per-pulse processing info
        WARNING: Skipped pulses, missing DMT data
        ERROR: Pulse processing failures
    """
    # Load spectrum once
    ext = Path(spectrum_file).suffix
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {ext}")
    
    data_class = SUPPORTED_EXTENSIONS[ext]
    spectrum = data_class(spectrum_file)
    header = spectrum.header()
    
    logger.info(f"Loaded spectrum: {spectrum_file}")
    freq_end_actual = header.fch1 + header.foff * (header.nchans - 1)
    logger.info(f"  Freq range: {header.fch1:.1f} - {freq_end_actual:.1f} MHz")
    
    generator = MultiViewImageGenerator(view_config)
    stats = {
        "total_pulses": len(pulses),
        "total_views": 0,
        "successful_views": 0,
        "failed_views": 0,
    }
    
    # Process each pulse
    for pulse_idx, pulse_data in enumerate(tqdm(pulses, desc="Processing pulses")):
        dm_inject, toa_inject, snr, width = pulse_data
        pulse_id = pulse_ids[pulse_idx] if pulse_ids else f"pulse_{pulse_idx:04d}"
        
        # Find segment for this pulse
        segment = generator._find_segment_for_dm(dm_inject)
        if segment is None:
            logger.warning(f"Pulse {pulse_id}: DM {dm_inject} not in any segment")
            continue
        
        logger.debug(f"Processing {pulse_id}: DM={dm_inject:.2f}, TOA={toa_inject:.4f}s")
        
        # Create pulse parameters
        pulse_params = PulseInjectionParams(
            dm_inject=dm_inject,
            toa_inject=toa_inject,
            snr=snr,
            width=width
        )
        
        # Generate all views
        views = generator.generate_all_views(
            spectrum,
            pulse_params, 
            pulse_id,
            segment,
            view_config.freq_start,
            view_config.freq_end
        )
        logger.debug(f"Generated {len(views)} views for {pulse_id}")
        
        # Save views
        for view_id, img in views:
            try:
                generator.save_view(view_id, img, output_dir)
                stats["successful_views"] += 1
            except Exception as e:
                logger.error(f"Failed to save {view_id}: {e}")
                stats["failed_views"] += 1
        
        stats["total_views"] += len(views)
    
    logger.info(f"Complete: {stats['successful_views']}/{stats['total_views']} views saved")
    return stats
