"""
Definition of enumerated types for time masking reasons. Used in hotpixel code and cosmic ray masking.
"""

import tables

timeMaskReasonList = []
timeMaskReasonList.append("unknown");
timeMaskReasonList.append("Flash in r0");
timeMaskReasonList.append("Flash in r1");
timeMaskReasonList.append("Flash in r2");
timeMaskReasonList.append("Flash in r3");
timeMaskReasonList.append("Flash in r4");
timeMaskReasonList.append("Flash in r5");
timeMaskReasonList.append("Flash in r6");
timeMaskReasonList.append("Flash in r7");
timeMaskReasonList.append("Flash in r8");
timeMaskReasonList.append("Flash in r9");
timeMaskReasonList.append("Merged Flash");
timeMaskReasonList.append("cosmic");
timeMaskReasonList.append("poofing");
timeMaskReasonList.append("hot pixel")
timeMaskReasonList.append("cold pixel")
timeMaskReasonList.append("dead pixel")
timeMaskReasonList.append("manual hot pixel")
timeMaskReasonList.append("manual cold pixel")
timeMaskReasonList.append("laser not on")       #Used in flashing wavecals
timeMaskReasonList.append("laser not off")      #Used in flashing wavecals
timeMaskReasonList.append("none")   #To be used for photon lists where the photon is NOT time-masked.

timeMaskReason = tables.Enum(timeMaskReasonList)


class TimeMask(tables.IsDescription):
    """The pytables derived class that stores time intervals to be masked"""
    tBegin = tables.UInt32Col() # beginning time of this mask (clock ticks)
    tEnd   = tables.UInt32Col() # ending time of this mask (clock ticks)
    reason = tables.EnumCol(timeMaskReason, "unknown", base='uint8')
