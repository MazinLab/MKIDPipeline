import mkidpipeline.config


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),)


HEADER_KEYS = tuple()


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.unstable', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


def fetch(data):
    pass