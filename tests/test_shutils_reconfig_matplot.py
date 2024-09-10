###Find a way to test if latex is present, then matplotlib is reconfigured to use it
###would test if shutils.which actively locates latex



###should return correct boolean values if shutils runs in the same way executible works

###NEXT:::  Ask and see how you can test if the use_latex function works and is incorportated into 
### the wavecal graphs



###NEXT:::also ask how to actually run test on the wavecall

import unittest
import shutil
import os 
import logging


from unittest import TestCase

class test_reconfig_matplot(TestCase):
    

    def test_reconfig(self):
        tex_installed = (shutil.which('latex') is not None and
                         shutil.which('dvipng') is not None and
                         shutil.which('ghostscript') is not None)
        
        
        latex_installed = (True if shutil.which('latex') is not None else False)
        dvipng_installed = (True if shutil.which('dving') is not None else False)
        ghost_installed = (True if shutil.which('ghostscript') is not None else False)
            
            
        list_to_check =(latex_installed, dvipng_installed,ghost_installed)

        for i in list_to_check:
            if i == True:
                self.assertTrue(i)
            else: 
                self.assertFalse(i)
       
       
       
       
       
       
       
        #if executables not on PATH
        #else:

            #log.warning("latex not configured to work with matplotlib")
 
 
 # add py project.toml add latex files a sdependencies, use try catch/
               
            
            

        





if __name__ == "__main__":

    unittest.main()


#####ideas###
#--use assert.true/false to verify that the next function runs if we use shutils.which()
#-- 













