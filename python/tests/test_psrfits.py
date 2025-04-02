import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
import your

from astroflow import dedispered_fil_with_dm
from astroflow.io.psrfits import PsrFits
from astroflow.io.data import Header


class TestPsrFits(unittest.TestCase):
    def setUp(self) -> None:
        fits_file_path = r"/data/QL/lingh/FAST_FRB_DATA/FRB20201124_0001.fits"
        self.filename = fits_file_path

    def test_psrfits_read(self) -> None:
        fits = PsrFits(self.filename)
