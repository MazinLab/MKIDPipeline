import unittest
from unittest import TestCase

class TestMKIDPipe(TestCase):
    def test_wavecal(self):
        import mkidpipeline.mkidpipe as mkidpipe
        import os
        os.chdir("./tests/data/hip109427-subset")
        self.assertEqual(mkidpipe.main(mkidpipe.parser().parse_args(args=["--make-dir", "--make-outputs"])), 0)

if __name__ == "__main__":
    unittest.main()
