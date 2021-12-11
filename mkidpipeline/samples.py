import pkg_resources
from collections import defaultdict

import mkidpipeline.definitions as definitions
import mkidpipeline.config as config

_i = defaultdict(lambda: 0)


def _namer(name='Thing'):
    ret = f"{name}{_i[name]}"
    _i[name] = _i[name] + 1
    return ret


SAMPLEDATA = {'default': (
    definitions.MKIDObservation(name=_namer('HIP109427'), start=1602048875, duration=10, wavecal='wavecal0',
                                dark=definitions.MKIDTimerange(name=_namer(), start=1602046500, duration=10),
                                flatcal='flatcal0', wcscal='wcscal0', speccal='speccal0'),
    # wavecals
    definitions.MKIDWavecal(name=_namer('wavecal'),
                            data=(definitions.MKIDTimerange(name='850 nm', start=1602040820, duration=60,
                                                                       dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                      duration=10),
                                                                       header=dict(laser='on', other='fits_key')),
                                             definitions.MKIDTimerange(name='950 nm', start=1602040895, duration=60,
                                                                       dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                      duration=10)),
                                             definitions.MKIDTimerange(name='1.1 um', start=1602040970, duration=60,
                                                                       dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                      duration=10)),
                                             definitions.MKIDTimerange(name='1.25 um', start=1602041040, duration=60,
                                                                       dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                      duration=10)),
                                             definitions.MKIDTimerange(name='13750 AA', start=1602041110, duration=60))
                            ),
    definitions.MKIDWavecal(name=_namer('wavecal'),
                            data=(definitions.MKIDTimerange(name='850 nm', start=1621333295, duration=60,
                                                                       header=dict(laser='on', other='fits_key')),
                                             definitions.MKIDTimerange(name='950 nm', start=1621333385, duration=60),
                                             definitions.MKIDTimerange(name='1.1 um', start=1621333505, duration=60),
                                             definitions.MKIDTimerange(name='1.25 um', start=1621333580, duration=60),
                                             definitions.MKIDTimerange(name='13750 AA', start=1621333770, duration=60))
                            ),
    # Flatcals
    definitions.MKIDFlatcal(name=_namer('flatcal'),
                            comment='Open dark that is being used for testing of white light flats while none are'
                                          ' available - should NOT be used as an actual flat calibration!',
                            data=definitions.MKIDObservation(name='open_dark', start=1576498833, duration=30.0,
                                                                        wavecal='wavecal0')),
    definitions.MKIDFlatcal(name=_namer('flatcal'), wavecal_duration=50.0, wavecal_offset=2.1, data='wavecal0'),
    # Speccal
    definitions.MKIDSpeccal(name=_namer('speccal'),
                            data=definitions.MKIDObservation(name=_namer('star'), start=1602049166, duration=10,
                                                                        wavecal='wavecal0',
                                                                        spectrum=pkg_resources.resource_filename('mkidpipeline', 'data/sample_spectrum.txt')),
                            aperture=('15h22m32.3', '30.32 deg', '200 mas')),

    # WCS cal
    definitions.MKIDWCSCal(name=_namer('wcscal'), pixel_ref=[107, 46], conex_ref=[-0.16, -0.4],
                           data='10.40 mas'),
    definitions.MKIDWCSCal(name=_namer('wcscal'),
                           comment='ob wcscals may be used to manually determine '
                                         'WCS parameters. They are not yet supported for '
                                         'automatic WCS parameter computation',
                           data=definitions.MKIDObservation(name=_namer('Sigma Ori'), start=1631279400,
                                                                       duration=20, wavecal='wavecal0',
                                                                       dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                      duration=10)),
                           pixel_ref=(107, 46), conex_ref=(0.0, 0.0),
                           source_locs=[['22h10m11.98527s', '+06d11m52.3017s'],
                                              ['22h11m11.98527s', '+06d11m52.3017s']]),
    # Dithers
    definitions.MKIDDither(name=_namer('dither'), data=1602047815, wavecal='wavecal0',
                           header=dict(OBJECT="HIP 109427"),
                           flatcal='flatcal0', speccal='speccal0', use='0,2,4-9', wcscal='wcscal0'),
    definitions.MKIDDither(name=_namer('dither'),
                           data=pkg_resources.resource_filename('mkidpipeline', 'data/dither_sample.log'),
                           wavecal='wavecal0', flatcal='flatcal0', speccal='speccal0', use=(1,),
                           wcscal='wcscal0'),
    definitions.MKIDDither(name=_namer('dither'), flatcal='', speccal='', wcscal='', wavecal='',
                           header=dict(OBJECT='HIP 109427'),
                           data=(definitions.MKIDObservation(name=_namer('HIP109427_'), start=1602047815,
                                                                        duration=10, wavecal='wavecal0', wcscal='wcscal0',
                                                                        header=dict(E_CONEXX=.1, E_CONEXY=-.4,
                                                                          OBJECT='HIP 109427'),
                                                                        dark=definitions.MKIDTimerange(name=_namer(), start=1602046500,
                                                                                                       duration=10)),
                                            definitions.MKIDObservation(name=_namer('HIP109427_'), start=1602047843, duration=10,
                                                                        wavecal='wavecal0', wcscal='wcscal0',
                                                                        header=dict(E_CONEXX=.1, E_CONEXY=-.28,
                                                                          OBJECT='HIP 109427')),
                                            definitions.MKIDObservation(name=_namer('HIP109427_'), start=1602047869,
                                                                        duration=10, wavecal='wavecal0', wcscal='wcscal0',
                                                                        header=dict(E_CONEXX=.1, E_CONEXY=-.16,
                                                                          OBJECT='HIP 109427'))
                                            )
                           )),
              'sigori': (
        definitions.MKIDObservation(name='Sigma Ori', start=1631279400, duration=10, wavecal='wavecal_sigori',
                                    dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10),
                                    flatcal='', wcscal='wcscal0', speccal=''),
        # wavecals
        definitions.MKIDWavecal(name='wavecal_sigori',
                                data=(definitions.MKIDTimerange(name='850 nm', start=1630603720, duration=60,
                                                                           dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10),
                                                                           header=dict(laser='on', other='fits_key')),
                                                 definitions.MKIDTimerange(name='950 nm', start=1630603840, duration=60,
                                                                           dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10)),
                                                 definitions.MKIDTimerange(name='1.1 um', start=1630603920, duration=60,
                                                                           dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10)),
                                                 definitions.MKIDTimerange(name='1.25 um', start=1630603995, duration=60,
                                                                           dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10)),
                                                 definitions.MKIDTimerange(name='13750 AA', start=1630604065,duration=60,
                                                                           dark=definitions.MKIDTimerange(name=_namer('dark'), start=1631273610, duration=10)))
                                ))
        }


def get_sample_data(dataset='default'):
    return SAMPLEDATA[dataset]


def get_sample_output(dataset='default'):
    data = [definitions.MKIDOutput(name=_namer('out'), data='dither0', min_wave='950 nm', max_wave='1375 nm', kind=k,
                                   duration=10.0) for k in ('drizzle', 'image', 'tcube', 'scube')]
    data.append(definitions.MKIDOutput(name=_namer('out'), data='dither0', min_wave='950 nm', max_wave='1375 nm',
                                       kind='drizzle', duration=10.0, timestep=1.0, wavestep= '0.0 nm'))
    data.append(definitions.MKIDOutput(name=_namer('out'), data='dither0', min_wave='950 nm', max_wave='1375 nm',
                                       kind='movie', duration=10.0, movie_format='gif', movie_runtime=3, movie_type='simple'))
    data.append(definitions.MKIDOutput(name=_namer('out'), data='dither0', min_wave='950 nm', max_wave='1375 nm',
                                       kind='movie', duration=10.0, movie_format='mp4', movie_runtime=3, movie_type='both'))
    return data
