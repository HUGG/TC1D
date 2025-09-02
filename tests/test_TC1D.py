#!/usr/bin/env python3
# Tests for tc1d

import pytest
import tc1d

class TestClass:
    def test_yr2sec(self):
        year_in_seconds = 31557600.0
        assert tc1d.yr2sec(1.0) == year_in_seconds

    def test_myr2sec(self):
        myr_in_seconds = 31557600000000.0
        assert tc1d.myr2sec(1.0) == myr_in_seconds

    def test_kilo2base(self):
        kilo = 1000.0
        assert tc1d.kilo2base(1.0) == kilo

    def test_milli2base(self):
        milli = 1.0e-3
        assert tc1d.milli2base(1.0) == milli

    def test_micro2base(self):
        micro = 1.0e-6
        assert tc1d.micro2base(1.0) == micro

    def test_mmyr2ms(self):
        mmyr_in_ms = 3.168808781e-11
        assert round(tc1d.mmyr2ms(1.0),20) == mmyr_in_ms

"""
class Test(unittest.TestCase):
    def test_yr2sec(self):
        year_in_seconds = 31557600.0
        self.assertEqual(year_in_seconds, TC1D.yr2sec(1.0))

    def test_myr2sec(self):
        myr_in_seconds = 31557600000000.0
        self.assertEqual(myr_in_seconds, TC1D.myr2sec(1.0))

    def test_kilo2base(self):
        kilo = 1000.0
        self.assertEqual(kilo, TC1D.kilo2base(1.0))

    def test_milli2base(self):
        milli = 1.0e-3
        self.assertEqual(milli, TC1D.milli2base(1.0))

    def test_micro2base(self):
        micro = 1.0e-6
        self.assertEqual(micro, TC1D.micro2base(1.0))

    def test_mmyr2ms(self):
        mmyr_in_ms = 3.168808781e-11
        self.assertEqual(mmyr_in_ms, round(TC1D.mmyr2ms(1.0),20))

if __name__ == '__main__':
    unittest.main()
"""