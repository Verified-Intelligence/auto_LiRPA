#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
class DummyGurobipyClass:
    """A dummy class with error message when gurobi is not installed."""
    def __getattr__(self, attr):
        def _f(*args, **kwargs):
            raise RuntimeError(f"method {attr} not available because gurobipy module was not built.")
        return _f

try:
    import gurobipy as grb
except ModuleNotFoundError:
    grb = DummyGurobipyClass()