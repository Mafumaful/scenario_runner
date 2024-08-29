"""
Cubic spline planner

Author: Mafumaful, Atsushi Sakai(@Atsushi_twi)

"""

import math
import numpy as np
import bisect

class CubicSpline1D:
    '''
    1D Cubic Spline class
    
    Parameters
    ----------
    x: list
        x coordinates for data points. This x coordinates must be sorted in ascending order.
    y: list
        y coordinates for data points
        
    '''
    
    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")
        
        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)
        
        # calculate the coefficients a
        self.a = [iy for iy in y]
        
        # calculate the coefficients c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        
        # calculate the coefficients b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i])\
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)
            
    def calc_position(self, x):
        """
        Calculate 'y' position for given 'x' position
        
        if 'x' is outside the data range, return None
        
        Returns
        -------
        y: float
            y position for given x position
        """
        
        if x<self.x[0] or x>self.x[-1]:
            return None
                
        i = self.__search_index(x)
        # check if run out of 
        if i == self.nx - 1:
            i = self.nx - 2
        
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3
        
        return position
    
    def calc_first_derivative(self, x):
        """
        Calculate the first derivative at x position
        
        if x is outside the data range, return None
        
        Returns
        -------
        dydx: float
            the first derivative at x position
        """
        
        if x<self.x[0] or x>self.x[-1]:
            return None
        
        i = self.__search_index(x)
        dx = x - self.x[i]
        dydx = self.b[i] + 2 * self.c[i] * dx + 3 * self.d[i] * dx ** 2
        
        return dydx
    
    def calc_second_derivative(self, x):
        """
        Calculate the second derivative at x position
        
        if x is outside the data range, return None
        
        Returns
        -------
        d2ydx2: float
            the second derivative at x position
        """
        
        if x<self.x[0] or x>self.x[-1]:
            return None
        
        i = self.__search_index(x)
        dx = x - self.x[i]
        d2ydx2 = 2 * self.c[i] + 6 * self.d[i] * dx
        
        return d2ydx2
    
    def __search_index(self, x):
        """
        search data segment index
        
        Returns
        -------
        i: int
            data segment index
        """
        
        return bisect.bisect(self.x, x) - 1
    
    def __calc_A(self, h):
        """
        calculate matrix A for spline coefficient c
        
        Returns
        -------
        A: numpy.ndarray
            matrix A for spline coefficient c
        """
        
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
            
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        
        return A
    
    def __calc_B(self, h):
        """
        calculate matrix B for spline coefficient c
        
        Returns
        -------
        B: numpy.ndarray
            matrix B for spline coefficient c
        """
        
        B = np.zeros(self.nx)
        
        for i in range(self.nx - 2):
            # h[i], h[i+1] must not be zero because of the assumption of the existence of the next data
            h[i] = 1e-6 if h[i] == 0 else h[i]
            h[i + 1] = 1e-6 if h[i + 1] == 0 else h[i + 1]
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1]\
                - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
            
        return B
        
class CubicSpline2D:
    """
    based on the cubic spline class, this class is for 2D cubic spline
    
    Parameters
    ----------
    x: list
        x coordinates for data points
    y: list
        y coordinates for data points
    """
    
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)
        
    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2) for idx, idy in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        
        return s
    
    def calc_position(self, s):
        """
        Calculate 'x, y' position for given 's' position
        
        if 's' is outside the data range, return None
        
        Returns
        -------
        x, y: float
            x, y position for given s position
        """
        
        if s<self.s[0] or s>self.s[-1]:
            return None
        
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)
        
        return x, y
    
    def calc_curvature(self, s):
        """
        Calculate curvature at s position
        
        if s is outside the data range, return None
        
        Returns
        -------
        curvature: float
            curvature at s position
        """
        
        if s<self.s[0] or s>self.s[-1]:
            print("s is out of range")
            return None
        
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        
        curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
        
        return curvature
    
    def calc_yaw(self, s):
        """
        Calculate yaw angle at s position
        
        if s is outside the data range, return None
        
        Returns
        -------
        yaw: float
            yaw angle at s position
        """
        
        if s<self.s[0] or s>self.s[-1]:
            return None
        
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        
        yaw = math.atan2(dy, dx)
        
        return yaw
    
def calc_spline_course(x, y, ds=0.1):
    sp  = CubicSpline2D(x, y)
    # print("sp.s[-1]: ", sp.s[-1])
    s = np.arange(0, sp.s[-1], ds)
    # print("lenth of s: ", len(s))
    
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
        
    # plot
    # import matplotlib.pyplot as plt
    # plt.plot(x, y, "xb", label="input")
    # plt.plot(rx, ry, "-r", label="spline")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()
        
    return rx, ry, ryaw, rk, sp
