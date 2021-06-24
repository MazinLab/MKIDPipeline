import multiprocessing as mp
import os
import threading
from collections import defaultdict
from contextlib import AbstractContextManager
import psutil
import time
from logging import getLogger

allocated = mp.Value('d', 0.0)
CHECK_INTERVAL = .5
MINFREE = 64*1024**3

print(f"======MEMORY PID={os.getpid()}======", allocated)


# With multiprocessing we need to ensure that the total amount of memory isn't gobbled up across all cores. The main
# hog is photontable use with python manages pretty well across threads within a process. What we'd like then is a way
# say with photontable.needed_ram(query='full'): ... that allows nesting. Nested calls don't need ram to stack (e.g.
# full followed by small would need no additional ram. So what we need for needed_ram() to return a ContextManager with
# an __enter__ method that waits for and reserves only the additional amount needed to reach the requested level for
# that file (on a per process basis) and and __exit__ that releases only the additional amount allocated


def get_free_ram():
    mem = psutil.virtual_memory()
    return mem.free + mem.cached - MINFREE


def reserve_ram(amount, id='', timeout=None):
    global allocated  # global across all processes
    elapsed = 0
    total = psutil.virtual_memory().total - MINFREE

    allocated.get_lock().acquire()
    available = min(get_free_ram(), total-allocated.value)

    while amount > available:
        allocated.get_lock().release()
        if timeout and elapsed > timeout:
            raise TimeoutError('Insufficient RAM available within timeout')
        if int(elapsed)%300 == 0:
            getLogger(__name__).debug(f'Waiting for {amount / 1024 ** 3:.1f} GB'+f' (for {id})' if id else '')
        time.sleep(CHECK_INTERVAL)
        elapsed += CHECK_INTERVAL

        allocated.get_lock().acquire()
        available = min(get_free_ram(), total - allocated.value)

    allocated.value += amount
    getLogger(__name__).debug(f'Reserved {amount / 1024 ** 3:.1f} GB, '
                              f'total reserved {allocated.value / 1024 ** 3:.1f} GB')
    allocated.get_lock().release()


def release_ram(amount, id=''):
    global allocated
    with allocated.get_lock():
        allocated.value -= amount
        getLogger(__name__).debug((f'{id} r' if id else 'R')+
                                  f'eleased {amount/1024**3:.1f} GB, total reserved {allocated.value/1024**3:.1f} GB')


class Manager(AbstractContextManager):

    def __init__(self, id):
        self.id=id
        self._lock = threading.Lock()
        self.required = {}
        self._allocated = defaultdict(list)

    def __setstate__(self, state):
        # don't preserve info across processes!
        self.__init__()

    def __call__(self, amount):
        self.required[threading.get_ident()] = amount
        return self

    def __enter__(self):
        tid=threading.get_ident()
        required = self.required[tid]
        # Some other thread might try to allocate simultaneously resulting in more than is required in total
        # we can fix this if needed but generally the pipeline is single threaded so ignore it for now
        with self._lock:  # small requests might get held up by a large request
            largest_allocation = self.required_allocation
            self._allocated[tid].append(required)
            if largest_allocation < required:
                reserve_ram(required-largest_allocation, id=f'{self.id} (tid={tid})')  # reserve a bit more
        return True

    def __exit__(self, exctype, excinst, exctb):
        """We should only release up to the incremental amount the corresponding enter call reserved.
        min(self._allocated-this_reservation, self._allocated - max(other_reservations))
        """
        tid = threading.get_ident()
        done_with = self._allocated[tid].pop()
        with self._lock:
            largest_required = self.required_allocation
        release_ram(max(done_with-largest_required, 0), id=f'{self.id} (tid={tid})')

    @property
    def required_allocation(self):
        try:
            return max(map(max, self._allocated.values()))
        except ValueError:
            return 0


#
# def lock_ram(amount, id, timeout=3600):
#     """ Repeated requests for a given id will ensure that the largest single request is reserved.
#     Requests do not stack, locks are per process"""
#
#     global allocated, allocations
#     elapsed = 0
#
#     if allocations.get(id, 0) >= amount:
#         return id
#
#     amount = amount - allocations.get(id, 0)
#
#     total = psutil.virtual_memory().total - MINFREE
#
#     free = get_free_ram()
#     allocated.get_lock().acquire()
#     available = min(free, total-allocated.value)
#
#     while amount > available:
#         allocated.get_lock().release()
#         if timeout and elapsed > timeout:
#             raise TimeoutError('Insufficient RAM available within timeout')
#         if int(elapsed)%300 == 0:
#             getLogger(__name__).debug(f'Waiting for {amount / 1024 ** 3:.1f} GB'+f' (for {id})' if id else '')
#         time.sleep(CHECK_INTERVAL)
#         elapsed += CHECK_INTERVAL
#
#         free = get_free_ram()
#         allocated.get_lock().acquire()
#         available = min(free, total - allocated.value)
#
#     allocations[id] = allocations.get(id,0) + amount
#     allocated.value += amount
#     allocated.get_lock().release()
#     with allocated.get_lock():
#         getLogger(__name__).debug(f'Reserved {amount/1024**3:.1f} GB, total reserved {allocated.value/1024**3:.1f} GB')
#
#     return id
#
#
# def unlock_ram(id):
#     global allocated, allocations
#     with allocated.get_lock():
#         allocations.pop(id)
#         allocated.value -= amount
#         getLogger(__name__).debug(f'Released {amount/1024**3:.1f} GB, total reserved {allocated.value/1024**3:.1f} GB')
