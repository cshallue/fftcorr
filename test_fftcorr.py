import unittest

import abc
import filecmp
import os
import shutil
import tempfile

import numpy as np

import fftcorr

NGRID = 256
MAX_SEP = 200.0	
DSEP = 10.0
MAX_ELL = 2
COSMOLOGY = {
    "omega": 0.317
}

DATA_DIR = "./test_data/"

# TODO: hack to make minimal changes to orignal code; refactor later.
fftcorr.BOSSpath = os.path.join(DATA_DIR, "lss/")


class BaseTest(abc.ABC, unittest.TestCase):
    """Abstract base class for running multiple configurations."""
    def setUp(self):
        # These are set in test() for each run.
        self.hemisphere = None
        self.periodic = None

    @property
    def ref_data_dir(self):
        """Returns the reference (ground truth) data directory."""
        return os.path.join(DATA_DIR, self.hemisphere,
                            "periodic" if self.periodic else "nonperiodic")

    @abc.abstractmethod
    def run_test(self):
        pass

    def test(self):
        """Runs run_test on all possible configurations."""
        for hemisphere in ["north", "south"]:
            for periodic in [True, False]:
                self.hemisphere = hemisphere
                self.periodic = periodic
                # TODO: make qperiodic nonglobal
                qperiodic_save = fftcorr.QPERIODIC
                fftcorr.QPERIODIC = self.periodic
                with self.subTest(hemisphere=hemisphere, periodic=periodic):
                    self.run_test()
                fftcorr.QPERIODIC = qperiodic_save


    def assertSameData(self, f1, f2, fmt):
        """Asserts that the numeric contents of f1 and f2 match closely."""
        if fmt == "bin":
            loader = np.fromfile
        elif fmt == "txt":
            loader = np.loadtxt
        else:
            raise ValueError("Unrecognized fmt: {}".format(fmt))

        f1_data = loader(f1)
        f2_data = loader(f2)
        np.testing.assert_allclose(f1_data,
                                   f2_data,
                                   err_msg="{} and {} do not match".format(
                                       os.path.basename(f1),
                                       os.path.basename(f2)))


class TestSetupCPP(BaseTest):
    def run_test(self):
        ref_dd_file = os.path.join(self.ref_data_dir, "corrDD.dat")
        ref_rr_file = os.path.join(self.ref_data_dir, "corrRR.dat")
        with tempfile.TemporaryDirectory() as test_dir:
            dd_file = os.path.join(test_dir, "corrDD.dat")
            rr_file = os.path.join(test_dir, "corrRR.dat")
            # TODO: make this a function in fftcorr.py.
            max_sep = 0.0 if self.periodic else MAX_SEP
            D, R = fftcorr.read_galaxies(
                self.hemisphere.title(), COSMOLOGY)
            grid = fftcorr.setup_grid(D, R, NGRID, max_sep)[1]
            fftcorr.writeCPPfiles(D, R, grid, dd_file, rr_file)
            self.assertSameData(dd_file, ref_dd_file, "bin")
            self.assertSameData(rr_file, ref_rr_file, "bin")


class TestCorrelateCPP(BaseTest):
    def run_test(self):
        ref_dd_infile = os.path.join(self.ref_data_dir, "corrDD.dat")
        ref_rr_infile = os.path.join(self.ref_data_dir, "corrRR.dat")
        ref_nn_outfile = os.path.join(self.ref_data_dir, "corrDD.dat.out")
        ref_rr_outfile = os.path.join(self.ref_data_dir, "corrRR.dat.out")
        with tempfile.TemporaryDirectory() as test_dir:
            # Copy the CPP inputs.
            # TODO: don't do this once the python code is refactored.
            dd_infile = os.path.join(test_dir, "corrDD.dat")
            rr_infile = os.path.join(test_dir, "corrRR.dat")
            shutil.copyfile(ref_dd_infile, dd_infile)
            shutil.copyfile(ref_rr_infile, rr_infile)
            # Run the CPP correlation code.
            fftcorr.correlateCPP(
                dd_infile, DSEP, NGRID, MAX_ELL, self.periodic, file2=rr_infile)
            fftcorr.correlateCPP(
                rr_infile, DSEP, NGRID, MAX_ELL, self.periodic)
            # Check outputs.
            self.assertSameData(dd_infile + ".out", ref_nn_outfile, "txt")
            self.assertSameData(rr_infile + ".out", ref_rr_outfile, "txt")


class TestComputeXi(BaseTest):
    # TODO: move this into fftcorr.py
    def load_cpp_out(self, filename):
        data = np.loadtxt(os.path.join(self.ref_data_dir, filename))
        data = data[data[:, 0] == 0]  # corr indicated by 0 in first col
        rcen = data[:, 1]
        hist_corr = data[:, 3:].T
        return hist_corr, rcen

    def run_test(self):
        ref_data = np.loadtxt(os.path.join(self.ref_data_dir, "xi.txt")).T
        ref_rcen = ref_data[0]
        ref_xi = ref_data[1:]

        hist_corrNN, rcen = self.load_cpp_out("corrDD.dat.out")
        hist_corrRR, _ = self.load_cpp_out("corrRR.dat.out")
        xi = fftcorr.analyze(hist_corrNN, hist_corrRR, rcen)[1]

        np.testing.assert_allclose(rcen, ref_rcen)
        np.testing.assert_allclose(xi, ref_xi, rtol=1e-5)


if __name__ == "__main__":
    del BaseTest  # Don't try to run BaseTest's test method.
    unittest.main()