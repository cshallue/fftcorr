import unittest

import abc
import filecmp
import os
import shutil
import tempfile

import numpy as np

import fftcorr

DSEP = 10.0

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
                qperiodic_save = fftcorr.qperiodic
                fftcorr.qperiodic = self.periodic
                with self.subTest(hemisphere=hemisphere, periodic=periodic):
                    self.run_test()
                fftcorr.qperiodic = qperiodic_save


class TestSetupCPP(BaseTest):
    def assertFilesEqual(self, f1, f2):
        """Asserts that the binary contents of f1 and f2 match exactly."""
        self.assertTrue(
            filecmp.cmp(f1, f2),
            "{} and {} do not match".format(os.path.basename(f1),
                                            os.path.basename(f2)))

    def run_test(self):
        ref_dd_file = os.path.join(self.ref_data_dir, "corrDD.dat")
        ref_rr_file = os.path.join(self.ref_data_dir, "corrRR.dat")
        with tempfile.TemporaryDirectory() as test_dir:
            dd_file = os.path.join(test_dir, "corrDD.dat")
            rr_file = os.path.join(test_dir, "corrRR.dat")
            # TODO: make this a function in fftcorr.py.
            if self.hemisphere == "north":
                D, R = fftcorr.read_dataNGC(fftcorr.cosmology)
            elif self.hemisphere == "south":
                D, R = fftcorr.read_dataSGC(fftcorr.cosmology)
            grid = fftcorr.setup_grid(D, R, fftcorr.max_sep)[1]
            fftcorr.writeCPPfiles(D, R, grid, dd_file, rr_file)
            self.assertFilesEqual(dd_file, ref_dd_file)
            self.assertFilesEqual(rr_file, ref_rr_file)


class TestCorrelateCPP(BaseTest):
    def assertSameData(self, f1, f2):
        """Assert that two text data files have the same data."""
        f1_data = np.loadtxt(f1)
        f2_data = np.loadtxt(f2)
        np.testing.assert_allclose(f1_data,
                                   f2_data,
                                   err_msg="{} and {} do not match".format(
                                       os.path.basename(f1),
                                       os.path.basename(f2)))

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
            fftcorr.correlateCPP(dd_infile,
                                 DSEP,
                                 file2=rr_infile)
            fftcorr.correlateCPP(rr_infile, DSEP)
            # Check outputs.
            self.assertSameData(dd_infile + ".out", ref_nn_outfile)
            self.assertSameData(rr_infile + ".out", ref_rr_outfile)


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