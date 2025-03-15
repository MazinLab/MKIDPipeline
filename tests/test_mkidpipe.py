import unittest
from unittest import TestCase

class TestMKIDPipe(TestCase):
    def test_init(self):
        import mkidpipeline.mkidpipe as mkidpipe
        import os
        from ruamel.yaml import YAML

        self.assertFalse(os.path.isfile("pipe.yaml"))
        self.assertFalse(os.path.isfile("data.yaml"))
        self.assertFalse(os.path.isfile("out.yaml"))

        self.assertEqual(mkidpipe.main(mkidpipe.parser().parse_args(args=["--init", "MEC"])), 0)
        self.assertTrue(os.path.isfile("pipe.yaml"))
        self.assertTrue(os.path.isfile("data.yaml"))
        self.assertTrue(os.path.isfile("out.yaml"))

        pipe, data, out = YAML(), YAML(), YAML()
        pipe.load(open("pipe.yaml", "r"))
        data.load(open("data.yaml", "r"))
        out.load(open("out.yaml", "r"))

        os.remove("pipe.yaml")
        os.remove("data.yaml")
        os.remove("out.yaml")

        self.assertEqual(mkidpipe.main(mkidpipe.parser().parse_args(args=["--init", "XKID"])), 0)
        self.assertTrue(os.path.isfile("pipe.yaml"))
        self.assertTrue(os.path.isfile("data.yaml"))
        self.assertTrue(os.path.isfile("out.yaml"))

        self.assertFalse(os.path.isfile("pipe_default.yaml"))
        self.assertFalse(os.path.isfile("data_default.yaml"))
        self.assertFalse(os.path.isfile("out_default.yaml"))

        pipe, data, out = YAML(), YAML(), YAML()
        pipe.load(open("pipe.yaml", "r"))
        data.load(open("data.yaml", "r"))
        out.load(open("out.yaml", "r"))

        os.remove("pipe.yaml")
        os.remove("data.yaml")
        os.remove("out.yaml")
        

if __name__ == "__main__":
    unittest.main()
