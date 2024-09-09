# Unit test for tqdm (used in lincal.py and wavecal.py)
# Note that I have seen the progress bar show up successfully, though only when manually changing verbose to verbose = True by default.

import unittest
import os

class TestTqdm(unittest.TestCase):
    
    def test_imports(self):
        import mkidpipeline.steps.wavecal
        import mkidpipeline.steps.lincal
        import tqdm

    def test_progressbar(self):
        # Test to see if progress bar shows up when passing data through
        os.chdir(r'/home/dnearing/pipelineconfig/targets/HIP109427')
        command1a = "echo 'BELOW IS THE PWD'"
        command1b = 'pwd'
        command1c = "echo 'ABOVE IS THE PWD'"
        command2 = 'mkidpipe --make-outputs'
        try:
        # Run the command
            os.system(command1a)
            os.system(command1b)
            os.system(command1c)
            os.system(command2)
            print("Command executed successfully.")
        except:
            print(f"An error occurred while running the command: {e}")



if __name__ == '__main__':
    unittest.main()