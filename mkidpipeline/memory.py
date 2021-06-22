import multiprocessing as mp
import psutil
import time
from logging import getLogger

allocated = mp.Value('d', 0.0)

CHECK_INTERVAL = .5
MINFREE = 32*1024**3


def get_free_ram():
    mem = psutil.virtual_memory()
    return mem.free + mem.cached - MINFREE


def lock_ram(amount, timeout=3600, id=''):

    global allocated
    elapsed = 0

    total = psutil.virtual_memory().total - MINFREE

    free = get_free_ram()
    allocated.get_lock().acquire()
    available = min(free, total-allocated.value)

    while amount > available:
        allocated.get_lock().release()
        if timeout and elapsed > timeout:
            raise TimeoutError('Insufficient RAM')
        if int(elapsed)%300 == 0:
            getLogger(__name__).debug(f'Waiting for {amount / 1024 ** 3:.1f} GB'+f' (for {id})' if id else '')
        time.sleep(CHECK_INTERVAL)
        elapsed += CHECK_INTERVAL

        free = get_free_ram()
        allocated.get_lock().acquire()
        available = min(free, total - allocated.value)

    allocated.value += amount
    allocated.get_lock().release()
    with allocated.get_lock():
        getLogger(__name__).debug(f'Reserved {amount/1024**3:.1f} GB, total reserved {allocated.value/1024**3:.1f} GB')
    return amount


def unlock_ram(amount):
    global allocated
    with allocated.get_lock():
        allocated.value -= amount
        getLogger(__name__).debug(f'Released {amount/1024**3:.1f} GB, total reserved {allocated.value/1024**3:.1f} GB')
