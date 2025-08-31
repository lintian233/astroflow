from .. import _astroflow_core as _astro_core #type: ignore

class IQRMConfig(_astro_core.IQRMconfig):
    def __init__(self, mode: int = 0, radius_frac: float = 0.0,
                 nsigma: float = 0.0, geofactor: float = 0.0, win_sec: float = 0.0,
                 hop_sec: float = 0.0, include_tail: bool = False):
        """
        Configuration for the IQRM (Iterative Quick RFI Mitigation) algorithm.

        Parameters:
        - mode (int): IQRM mode (0 for mean, 1 for median).
        - raise_factor (float): Raise factor for thresholding.
        - radius_frac (float): Radius as a fraction of total channels.
        - nsigma (float): Number of standard deviations for thresholding.
        - geofactor (float): Geometric factor for RFI detection.
        - win_sec (float): Window size in seconds.
        - hop_sec (float): Hop size in seconds.
        - include_tail (bool): Whether to include tail channels in processing.
        """
        super().__init__(mode, radius_frac,
                         nsigma, geofactor, win_sec,
                         hop_sec, include_tail)



class RFIConfig(_astro_core.RFIconfig):
    def __init__(self, use_mask: bool, use_zero_dm: bool, use_iqrm: bool, iqrm_cfg = None):
        """
        RFI configuration for data processing.

        Parameters:
        - use_mask (bool): Whether to apply a mask file for RFI mitigation.
        - use_zero_dm (bool): Whether to apply zero-DM filtering.
        - use_iqrm (bool): Whether to apply the IQRM algorithm for RFI mitigation.
        - iqrm_cfg (IQRMConfig or None): Configuration for IQRM algorithm. If None, IQRM is not applied.
        """
        if use_iqrm and iqrm_cfg is None:
            iqrm_cfg = _astro_core.IQRMconfig(
                0,      # mode
                0.1,    # radius_frac
                10.0,    # nsigma
                1.5,    # geofactor
                0,    # win_sec
                2.5,    # hop_sec
                True    # include_tail
            )
        super().__init__(use_mask, use_zero_dm, use_iqrm, iqrm_cfg)