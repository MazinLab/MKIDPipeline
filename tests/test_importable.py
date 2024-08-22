import unittest
from unittest import TestCase

class TestImportable(TestCase):
    def test_imports(self):
        import mkidpipeline
        import mkidpipeline.definitions
        import mkidpipeline.pipeline
        import mkidpipeline.config
        import mkidpipeline.steps
        import mkidpipeline.samples

if __name__ == "__main__":
    unittest.main()
