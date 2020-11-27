import unittest

import os
import tempfile
import shlex
import subprocess

import numpy as np

DATA_DIR = "./test_data/abacus/smallmass/"


def read_outfile(filename):
    data = np.loadtxt(filename)
    corr = data[data[:, 0] == 0]  # corr indicated by 0 in first col
    return corr


# TODO: test power spectrum, nonperiodic.
class TestCorrelateCPP(unittest.TestCase):
    def test_correlate_cpp(self):
        # TODO: rename these from DD?
        infile = os.path.join(DATA_DIR, "corrDD.dat")
        ref_outfile = os.path.join(DATA_DIR, "corrDD.dat.out")
        with tempfile.TemporaryDirectory() as test_dir:
            outfile = os.path.join(test_dir, "corrDD.dat.out")
            cmd = ("{}/fftcorr -in {} -out {} -dr 5 -n 256 -p -r 250.00 -iso "
                   "-normalize -w 0").format(os.getcwd(), infile, outfile)
            print(cmd)
            self.assertEqual(subprocess.call(shlex.split(cmd)), 0)

            corr = read_outfile(outfile)
            ref_corr = read_outfile(ref_outfile)
            np.testing.assert_allclose(
                corr,
                ref_corr,
                err_msg="Correlation function does not match {}".format(
                    ref_outfile))


if __name__ == "__main__":
    unittest.main()