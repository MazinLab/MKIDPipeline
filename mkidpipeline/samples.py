import pkg_resources
from collections import defaultdict
import mkidpipeline.config as config
i = defaultdict(lambda: 0)


def namer(name='Thing'):
    ret = f"{name}{i[name]}"
    i[name] = i[name] + 1
    return ret


SAMPLEDATA = {'default': (
    config.MKIDObservation(name=namer('star'), start=1602048875, duration=10, wavecal='wavecal0',
                           dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10),
                           flatcal='flatcal0', wcscal='wcscal0', speccal='speccal0'),
    # a wavecal
    config.MKIDWavecalDescription(name=namer('wavecal'),
                                  data=(config.MKIDTimerange(name='850 nm', start=1602040820, duration=60,
                                                             dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                       duration=10),
                                                             header=dict(laser='on', other='fits_key')),
                                        config.MKIDTimerange(name='950 nm', start=1602040895, duration=60,
                                                             dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                       duration=10)),
                                        config.MKIDTimerange(name='1.1 um', start=1602040970, duration=60,
                                                             dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                       duration=10)),
                                        config.MKIDTimerange(name='1.25 um', start=1602041040, duration=60,
                                                             dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                       duration=10)),
                                        config.MKIDTimerange(name='13750 AA', start=1602041110, duration=60))
                                  ),
    # Flatcals
    config.MKIDFlatcalDescription(name=namer('flatcal'),
                                  data=config.MKIDObservation(name='950 nm', start=1602040900, duration=50.0,
                                                              dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                        duration=10),
                                                              wavecal='wavecal0')),
    config.MKIDFlatcalDescription(name=namer('flatcal'), wavecal_duration=50.0, wavecal_offset=2.1, data='wavecal0'),
    # Speccal
    config.MKIDSpeccalDescription(name=namer('speccal'),
                                  data=config.MKIDObservation(name=namer('star'), start=1602049166, duration=10,
                                                              wavecal='wavecal0', spectrum='qualified/path/or/relative/'
                                                                                           'todatabase/refspec.file'),
                                  aperture=('15h22m32.3', '30.32 deg', '200 mas')),

    # WCS cal
    config.MKIDWCSCalDescription(name=namer('wcscal'), dither_home=[107, 46], dither_ref=[-0.16, -0.4], #TODO UPDATE to MEC KEYS
                                 data='10.40 mas'),
    config.MKIDWCSCalDescription(name=namer('wcscal'),
                                 comment='ob wcscals may be used to manually determine '
                                         'WCS parameters. They are not yet supported for '
                                         'automatic WCS parameter computation',
                                 data=config.MKIDObservation(name=namer('star'), start=1602047935,
                                                             duration=10, wavecal='wavecal0',
                                                             dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                       duration=10)),
                                 dither_home=(107, 46), dither_ref=(-0.16, -0.4)),  #TODO UPDATE to MEC KEYS
    # Dithers
    config.MKIDDitherDescription(name=namer('dither'), data=1602047815, wavecal='wavecal0',
                                 flatcal='flatcal0', speccal='speccal0', use='0,2,4-9', wcscal='wcscal0'),
    config.MKIDDitherDescription(name=namer('dither'),
                                 data=pkg_resources.resource_filename('mkidpipeline', 'dither_sample.log'),
                                 wavecal='wavecal0', flatcal='flatcal0', speccal='speccal0', use=(1,),
                                 wcscal='wcscal0'),
    config.MKIDDitherDescription(name=namer('dither'), flatcal='', speccal='', wcscal='', wavecal='',
                                 header=dict(OBJECT='HIP 109427'),
                                 data=(config.MKIDObservation(name=namer('HIP109427_'), start=1602047815,
                                                              duration=10, wavecal='wavecal0',
                                                              header=dict(M_CONEXX=.2, M_CONEXY=.3,
                                                                          OBJECT='HIP 109427'),
                                                              dark=config.MKIDTimerange(name=namer(), start=1602046500,
                                                                                        duration=10)),
                                       config.MKIDObservation(name=namer('HIP109427_'), start=1602047825, duration=10,
                                                              wavecal='wavecal0', header=dict(M_CONEXX=.1, M_CONEXY=.1),
                                                              wcscal='wcscal0'),
                                       config.MKIDObservation(name=namer('HIP109427_'), start=1602047835,
                                                              duration=10, wavecal='wavecal0', wcscal='wcscal0',
                                                              header=dict(M_CONEXX=-.1, M_CONEXY=-.1))
                                       )
                                 )
)}
