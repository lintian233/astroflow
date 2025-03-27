import numpy as np
import unittest
import os
import matplotlib.pyplot as plt

from astroflow import dedispered_fil_with_dm
from astroflow import Spectrum, Filterbank


class TestDedispered(unittest.TestCase):
    def test_dedispered_fil_with_dm_uint8(self):
        fil = Filterbank("../tests/FRB180417.fil")
        tstart = 0
        tend = 2
        dm = 474.14
        spectrum = dedispered_fil_with_dm(fil, tstart, tend, dm)
        assert isinstance(spectrum, Spectrum)
        return
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            dpi=100,
            gridspec_kw={"height_ratios": [1, 3]},
            sharex=True,
        )
        plt.rcParams["image.origin"] = "lower"

        vmin, vmax = np.percentile(spectrum.data, [1, 99])
        data = spectrum.data
        time_axis = np.linspace(tstart, tend, spectrum.ntimes)
        freq_axis = fil.fch1 + np.arange(spectrum.nchans) * fil.foff
        time_series = data.sum(axis=1)
        axs[0].plot(time_axis, time_series, "k-", linewidth=0.5)
        axs[0].set_ylabel("Integrated Power")
        axs[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axs[0].set_yscale("log")
        axs[0].grid(True, alpha=0.3)

        # 设置imshow的显示范围和方向
        extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        axs[1].imshow(
            data.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axs[1].set_ylabel(
            f"Frequency (MHz)\nFCH1={fil.fch1:.3f} MHz, FOFF={fil.foff:.3f} MHz"
        )
        axs[1].set_xlabel(f"Time (s)\nTSAMP={fil.tsamp:.6e}s")

        axs[0].set_xlim(tstart, tend)
        axs[1].set_xlim(tstart, tend)

        plt.subplots_adjust(hspace=0.05, left=0.08, right=0.92)
        plt.savefig(
            "test_dedispered_fil_with_dm_uint8.png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            format="png",
            pil_kwargs={"compress_level": 0},
        )
        plt.close()
