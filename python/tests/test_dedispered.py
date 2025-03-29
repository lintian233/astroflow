import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
import your

from astroflow import dedispered_fil_with_dm
from astroflow import Spectrum, Filterbank


class TestDedispered(unittest.TestCase):
    def test_dedispered_fil_with_dm(self) -> None:
        test_file = r"../tests/qltest.fil"
        if not os.path.exists(test_file):
            test_file = r"../tests/FRB180417.fil"

        filterbank = Filterbank(test_file)

        start_time = 0.0  # 起始时间(秒)
        end_time = 0.5  # 结束时间(秒)
        test_dm = 0.0  # 测试用色散量

        processed_spectrum = dedispered_fil_with_dm(
            filterbank, start_time, end_time, test_dm
        )
        your_reader = your.Your(test_file)

        header = your_reader.your_header
        tsamp = header.tsamp  # 采样时间(秒)
        total_samples = int((end_time - start_time) / tsamp)

        your_raw_data = your_reader.get_data(start_time * tsamp, total_samples)

        if header.foff < 0:
            your_raw_data = your_raw_data[:, ::-1]  # 反转频率轴

        filterbank_slice = filterbank.data[:total_samples, 0, :]

        self.assertIsInstance(processed_spectrum, Spectrum)

        self.assertTrue(
            np.allclose(
                processed_spectrum.data[:, :],  # 处理后的数据
                your_raw_data[:, :-1],  # your库数据（排除最后一个频率通道）
                atol=1e-6,
            ),
            "频谱数据与your库不一致",
        )

        self.assertTrue(
            np.allclose(filterbank_slice, your_raw_data, atol=1e-6),
            "原始数据切片不一致",
        )
