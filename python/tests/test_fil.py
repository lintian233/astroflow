import _astroflow_core as astroflow  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import unittest
import os
import your

from _astroflow_core import Filterbank  # type: ignore


class TestFilterbank(unittest.TestCase):
    def test_fil_read(self):
        test_file = r"../tests/qltest.fil"
        if not os.path.exists(test_file):
            test_file = r"../tests/FRB180417.fil"

        filterbank = Filterbank(test_file)

        start_time = 0.0
        end_time = 0.5

        your_reader = your.Your(test_file)

        header = your_reader.your_header

        tsamp = header.tsamp
        total_samples = int((end_time - start_time) / tsamp)

        your_raw_data = your_reader.get_data(start_time * tsamp, total_samples)

        if header.foff < 0:
            your_raw_data = your_raw_data[:, ::-1]  # 反转频率轴

        # only support 1 polarization data
        filterbank_slice = filterbank.data[:total_samples, 0, :]

        self.assertTrue(
            np.allclose(filterbank_slice, your_raw_data, atol=1e-6),
        )
