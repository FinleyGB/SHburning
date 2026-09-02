"""
Backwards-compatible shim.

The simulation code now lives in ``shb.py``. This module is kept so the older
notebook ``SHsimulation_v1_3test2multi.ipynb`` still runs with its
``from worker import *`` line.

The single-thread BLAS environment variables are set here, before numpy is
imported anywhere, because that is what keeps a multiprocessing.Pool scan from
oversubscribing the cores. ``shb.py`` now sets the same block itself, ahead of
its own numpy import, so scripts and the GUI get it too; this copy stays
because a notebook importing this module has to get in before numpy either way.
``setdefault`` makes the duplication harmless.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from shb import *  # noqa: F401,F403,E402
from shb import DEFAULT_PARAMS  # noqa: E402

# The notebooks build their own ``params`` dict in the config cell; expose the
# default here too for convenience.
params = DEFAULT_PARAMS
