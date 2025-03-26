import _astroflow_core as astroflow  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import unittest

from _astroflow_core import Filterbank  # type: ignore


class TestFilterbank(unittest.TestCase):
    def test_read_fil(self):
        fil = Filterbank("../tests/FRB180417.fil")
        self.assertEqual(fil.data[:, 0, :].shape, (fil.ndata, fil.nchans))
        raw_data = fil.data
        assert isinstance(raw_data, np.ndarray)
        assert raw_data.dtype == np.uint8
        assert raw_data.shape == (fil.ndata, fil.nifs, fil.nchans)

        return

        vmin, vmax = np.percentile(raw_data, [1, 99])

        x_shape = raw_data.shape[0]
        y_shape = raw_data.shape[2]
        plt.figure(figsize=(20, 8), dpi=100)
        plt.rcParams["image.origin"] = "lower"

        tstart = 2.0
        tend = 3.0
        start_idx = int(tstart / fil.tsamp)
        end_idx = int(tend / fil.tsamp)
        t_len = end_idx - start_idx
        time_axis = np.linspace(tstart, tend, t_len)
        freq_axis = fil.fch1 + np.arange(y_shape) * fil.foff
        print(
            f"t_len: {t_len}, time_axis: {time_axis.shape}, freq_axis: {freq_axis.shape}"
        )
        print(f"raw_data: {raw_data.shape}")
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        print(
            f"raw_data[start_idx:end_idx, 0, :]: {raw_data[start_idx:end_idx, 0, :].shape}"
        )
        plt.imshow(
            raw_data[start_idx:end_idx, 0, :].T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlabel(f"Time (s)\nTSAMP={fil.tsamp:.6e}s")
        plt.ylabel(f"Frequency (MHz)\nFCH1={fil.fch1:.3f} MHz, FOFF={fil.foff:.3f} MHz")

        plt.savefig(
            "test_fil.png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            format="png",
            pil_kwargs={"compress_level": 0},
        )  # Disable compression

        plt.close()
