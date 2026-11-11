"""Module used to solve the Bateman's Equation.

This is a fancy name for a time-dependent matrix equation:

dn(t)/dt = A(t)n(t)

This module solves that equation using approximations to e^(At), as the actual
matrix exponential is numerically unstable. This is because some isotopes decay
very quickly and some decay very slowly, at very different time scales.

It is common to solve these using CRAM (Chebychev Rational Approximation
Method), though other methods exist as well.

As always, the actual numerical heart is vastly easier than the utilities
needed to generate the numerical data from nuclear databases and user input.

"""

from .graphs import DecayGraph as DecayGraph, ReactionGraph as ReactionGraph, GraphFilter as GraphFilter
from .solver import *

jsonable = ["DecayGraph"]
