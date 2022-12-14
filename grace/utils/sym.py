"""Compute the intersection of two gaussians
How to use:
>>> G = GaussianModel(m1, m2)
>>> G.value

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""

from __future__ import division
from sympy import symbols, solve, Eq, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
import argparse


from grace.utils import Logger
log = Logger(__name__).logger

from decimal import *
getcontext().prec = 6

x = symbols('x')
FLOAT_PRECISION = 10


class GaussianModel:
    def __init__(self, mean1, mean2, method="closed"):
        """_summary_

        Args:
            mean1 (_type_): mean of first gaussian
            mean2 (_type_): mean of second gaussian
            method (str, optional): Method for computing the intersection, "closed" or "standard". Defaults to "standard".
        """
        self.m1 = round(mean1, FLOAT_PRECISION)
        self.m2 = round(mean2, FLOAT_PRECISION)
        log.debug(f'm1: {self.m1}, m2: {self.m2}')
        self.f1 = exp(-(x + self.m1)**2 / 2)
        self.f2 = exp(-(x + self.m2)**2 / 2)

        self.itx = None
        self.ity = None
        
        self.same = False
        if np.isclose(self.m1, self.m2):
            self.same_means()

        self.file_path = None

        try:
            if self.itx is None or self.ity is None:
                if method == "closed":
                    self.ity = self.closed_form()
                elif method == "standard":
                    self.itx = self.standard_form()
                    log.debug(f'itx: {self.itx}')
                    self.ity = self.get_y_val()
                else:
                    raise ValueError("method must be 'closed' or 'standard'")
            log.info(f'value: {self.ity}')
            self.value = self.ity
            self.precision()
        except Exception as e:
            log.fatal(f'Error: {e}')

    def same_means(self):
        self.itx = self.m1
        self.ity = 1.0
        self.same = True

    def standard_form(self):
        eq = Eq(self.f1, self.f2)
        intersection = solve(eq, dict=True, set=True)
        if len(intersection) == 1:
            itx = intersection[0][x]
        else:
            log.debug(f'intersection: {intersection}')

        log.debug(f'standard itx: {itx}')
        return itx

    def closed_form(self):
        closed_form = (self.m1**2 - self.m2**2) / (2 * (self.m1 - self.m2))
        self.itx = -closed_form
        f = exp(-1 / 2 * (x - self.m1)**2)
        ity = f.subs(x, closed_form)
        log.debug(f'closed ity: {ity}')
        if ity < 1e-10:
            ity = 0.0
        return round(ity, FLOAT_PRECISION)

    def get_y_val(self):
        try:
            subs = self.f1.subs(x, self.itx)
            # log.debug(subs)
            ity = subs.evalf()
        except Exception as e:
            log.warning(e)
        return ity

    def precision(self):
        self.itx = round(float(self.itx), FLOAT_PRECISION)
        self.ity = round(float(self.ity), FLOAT_PRECISION)
        self.value = self.ity

    def plot_once(self, type, path=None):
        point_of_intersection = [self.itx, self.ity]
        xx = np.linspace(-10 + int(self.itx), 10 + int(self.itx), 1000)
        yy = lambdify(x, [self.f1, self.f2])(xx)
        plt.plot(xx, np.transpose(yy))
        plt.scatter(*point_of_intersection)
        self.file_path = path + '/gauss_result.png'
        if type == "screen":
            plt.show()
        elif type == "file":
            plt.savefig(self.file_path)
        else:
            plt.savefig(self.file_path)
            plt.show()

    def plot_continuous(self):
        pass

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, GaussianModel):
            return NotImplemented
        return self.itx == __o.itx and self.ity == __o.ity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('mean1',
                        metavar='M1',
                        type=float,
                        nargs='+',
                        help='mean for the first gaussian')
    parser.add_argument('mean2',
                        metavar='M2',
                        type=float,
                        nargs='+',
                        help='mean for the second gaussian')

    parser.add_argument(
        '--plot',
        choices=['screen', 'file', 'screen_and_file'],
        help='Show result either screen, file or screen_and_file')

    # Parse and log.debug the results
    args = parser.parse_args()
    m1 = args.mean1[0]
    m2 = args.mean2[0]
    plot_type = args.plot
    log.debug(f'mean1: {m1}')
    log.debug(f'mean2: {m2}')
    G = GaussianModel(m1, m2)

    # G.plot_once(type=plot_type)
