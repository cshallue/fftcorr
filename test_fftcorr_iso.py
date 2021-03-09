import unittest

import os
import tempfile
import shlex
import subprocess

import numpy as np

DATA_DIR = "./test_data/abacus/smallmass/"


def read_outfile(filename):
    data = np.loadtxt(filename)
    pspec = data[data[:, 0] == 1]
    corr = data[data[:, 0] == 0]
    return pspec, corr


class TestCorrelateCPP(unittest.TestCase):
    def test_correlate_cpp(self):
        # TODO: rename these from DD?
        infile = os.path.join(DATA_DIR, "corrDD.dat")
        ref_outfile = os.path.join(DATA_DIR, "corrDD.dat.out")
        with tempfile.TemporaryDirectory() as test_dir:
            outfile = os.path.join(test_dir, "corrDD.dat.out")
            cmd = ("{}/cc/fftcorr -in {} -out {} -n 256 -p -r 250.00 -dr 5 "
                   "-kmax 0.4 -dk 0.002 -maxell 0 -w 0 -periodic").format(
                       os.getcwd(), infile, outfile)
            print(cmd)
            self.assertEqual(subprocess.call(shlex.split(cmd)), 0)

            pspec, corr = read_outfile(outfile)
            ref_pspec, ref_corr = read_outfile(ref_outfile)
            np.testing.assert_allclose(pspec,
                                       ref_pspec,
                                       err_msg="Power spectrum does not match")
            np.testing.assert_allclose(
                corr, ref_corr, err_msg="Correlation function does not match")


if __name__ == "__main__":
    unittest.main()