from os.path import exists
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
import your
import time

from astroflow import dedispered_fil_with_dm
from astroflow.dedispered import (
    dedisperse_spec,
    dedisperse_spec_with_dm,
)
from astroflow import Spectrum, Filterbank
from astroflow.io.data import Header
from astroflow.io.psrfits import PsrFits
from astroflow.plotter import PlotterManager


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

        filterbank_slice = filterbank.get_spectrum()[:total_samples, :]

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

    def test_dedisperse_all_spec_with(self) -> None:
        test_file = r"../tests/qltest.fil"
        if not os.path.exists(test_file):
            test_file = r"../tests/FRB180417.fil"

        filterbank = Filterbank(test_file)
        data = filterbank.get_spectrum()
        header = filterbank.header()
        dm = 0

        freq_start = 1250
        freq_end = 1430
        t_sample = 0.5

        freq_start_chan = int((freq_start - header.fch1) / header.foff)
        freq_end_chan = int((freq_end - header.fch1) / header.foff)
        print(f"Frequency start channel: {freq_start_chan}")
        print(f"Frequency end channel: {freq_end_chan}")

        spec = dedisperse_spec_with_dm(
            filterbank,
            0,
            header.tsamp * header.ndata,
            dm,
            freq_start=freq_start,
            freq_end=freq_end,
        )

        self.assertIsInstance(spec, Spectrum)

        self.assertTrue(
            np.allclose(
                spec.data[:, :],
                data[:, freq_start_chan:freq_end_chan],
                atol=1e-6,
            ),
            "频谱数据与原始数据不一致",
        )
