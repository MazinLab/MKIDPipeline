import mkidpipeline.config
from mkidcore.pixelflags import FlagSet


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!samplestep_cfg'
    REQUIRED_KEYS = (('parameter1', 2, 'comment about parameter1, list allowed values here'),
                     ('parameter2', 'string val', 'comment about parameter2')
                     )

    def _vet_errors(self):
        ret = []
        """
        Here is where you can assert that the keys in your config have values within allowed ranges or of allowed types.
        This will help to catch errors with the config at the beginning of running the pipeline if 'vet' is specified.
        """
        return ret


FLAGS = FlagSet.define(('flag1', 1, 'description of flag1'),
                       ('flag2', 2, 'description of flag2'),
                       ('flag3', 3, 'description of flag3'))

PROBLEM_FLAGS = tuple()  # Here is where you will define flags set by other pipeline steps that make a pixel incompatible
# with your step e.g. 'beammap.noDacTone'

"""
The code to perform your calibration step goes here. It can be composed of functions, classes, you name it! It just
needs to be accessed by fetch (below). 
"""

def fetch(dataset, config=None):
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(yourstep=StepConfig()), cfg=config, copy=True)
    """
    Needs to return or save a calibration product that 'apply' can use to apply it to the h5 file. This is the function
    that will access your step specific code (above). fetch will often require a relevant dataset to run on. 
    """


def apply(o, config=None):
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(yourstep=StepConfig()), cfg=config, copy=True)
    """
    apply needs to apply the results of your step (from fetch) to each h5 file ('o') and also update relevant fields
    in the header and apply the FLAGS corresponding to your step 
    """
