from pylab import *
import math
import time

## note: even though using pylab, still need to specify plt.plot() becuase plot() may
## conflict with a sympy function                                                                                                                                                  

class LineSpline:
    def __init__(self, start, end):
        self.s = start
        self.e = end

    def closestXGivenY(self, vts, vts_h):
        # linear function t*s + (1-t)*e = vt_i
        # solve for t given y-coords, find closest x-coord

        s = array([self.s]*len(vts)).astype(float)
        e = array([self.e]*len(vts)).astype(float)

        isHorizLine = s[:,1]==e[:,1]

        a = s[:,1] - e[:,1]
        b = e[:,1] - vts[:,1]
        t = -1*b/a

        isSSmaller = s[:,0]<e[:,0]
        isESmaller = s[:,0]>=e[:,0]
        t[isHorizLine*isSSmaller] = zeros((sum(isHorizLine*isSSmaller)))
        t[isHorizLine*isESmaller] = ones((sum(isHorizLine*isESmaller)))

        closest_xs = array([Inf]*len(vts)).astype(complex)

        isValid = (.00001<=t)*(t<=1.00001)
        x_ts = t*s[:,0] + (1-t)*e[:,0]
        
        spline_h = self.getSplineHash()
        x_tsh = array([i+spline_h for i in x_ts.astype(str_)]).astype(str_) #calculated points' hash array
        isNotSame = x_tsh!=vts_h
        isClosest = (x_ts-vts[:,0]>0)*isValid*isNotSame
        closest_xs[isClosest] = x_ts[isClosest]

        '''
        for t in ts:
            if (arg(t)  == 0 or math.isnan(arg(t))) and (t >= 0 and t <= 1):
                x_t = t*s[0] + (1-t)*e[0]
                x_th = str(x_t) + self.getSplineHash()
                
                if x_t - vt_i[0] > 0 and not x_th == vt_ih:
                    closest_x = x_t
        '''

        return [closest_xs, x_tsh]

    def smallestX(self):
        return min(self.s[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.e[1])

    def biggestY(self):
         return max(self.s[1], self.e[1])

    def getSplineHash(self):
        s = self.s
        e = self.e

        return 'lin'+str(s[0])+str(s[1])+str(e[0])+str(e[1])
    
    def plotSpline(self):
        s = self.s
        e = self.e

        params = array(range(101))/100.0
        sp_pts = []
        for t in params: #based on the formula for linear splines
            x_t = float(t*s[0] + (1-t)*e[0])
            y_t = float(t*s[1] + (1-t)*e[1])
            sp_pts.append([x_t,y_t])

        sp_pts = array(sp_pts)
        plt.plot(sp_pts[:,0], sp_pts[:,1], 'g-', linewidth=3)

class QuadBezSpline:
    def __init__(self, start, ctl, end):
        self.s = start
        self.c = ctl
        self.e = end

    def closestXGivenY(self, vts, vts_h):
        # quadtric function a*t^2 + b*t + c = 0, using vt_iy
        # solve for t given a,b,c, use it to find closest x to the right of vt
        # coeffs are derived from the quad bez formua

        s = array([self.s]*len(vts)).astype(complex)
        c = array([self.c]*len(vts)).astype(complex)
        e = array([self.e]*len(vts)).astype(complex)

        a = s[:,1] - 2*c[:,1] + e[:,1]
        b = 2*c[:,1] - 2*s[:,1]
        c = s[:,1] - vts[:,1]
        t1 = (-b + (b*b - 4*a*c)**.5)/(2*a)
        t2 = (-b - (b*b - 4*a*c)**.5)/(2*a)
        ts = [t1, t2]

        closest_xs = array([Inf]*len(vts)).astype(complex)
        closest_xsh = array(['.']*len(vts)).astype(str_)
        for t in ts:
            isValid = (.00001<=t)*(t<=1.00001)
            x_ts = ((1-t)**2)*s[:,0] + 2*(1-t)*t*c[:,0] + t*t*e[:,0]

            spline_h = self.getSplineHash()
            x_tsh = array([i+spline_h for i in x_ts.astype(str_)]).astype(str_) #calculated points' hash array
            isNotSame = x_tsh!=vts_h
            isClosest = (x_ts < closest_xs)*(x_ts-vts[:,0]>0)*isValid*isNotSame
            closest_xs[isClosest] = x_ts[isClosest]
            closest_xsh[isClosest] = x_tsh[isClosest]

        closest_xs[closest_xs==Inf] = array(['.']*sum(closest_xs==Inf)).astype(str_)
            
        '''
        closest_x = '.'
        for t in ts:
            if (arg(t) == 0 or math.isnan(arg(t))) and (t >= 0 and t <= 1): #checks to make sure real root and is valid param val
                x_t = s[0]*pow(1-t,2) + c[0]*(2*t)*(1-t) + e[0]*pow(t,2)

                x_th = str(x_t)+self.getSplineHash()

                if x_t - vt_i[0] > 0 and closest_x == '.' and not x_th == vt_ih:
                    closest_x = x_t
                elif x_t - vt_i[0] > 0 and x_t < closest_x and not x_th == vt_ih: #ensures closest pt isn't vt_ix
                    closest_x = x_t
        '''
        return [closest_xs, closest_xsh]

    def smallestX(self):
        return min(self.s[0], self.c[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.c[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.c[1], self.e[1])

    def biggestY(self):
        return max(self.s[1], self.c[1], self.e[1])

    def getSplineHash(self):
        s = self.s
        c = self.c
        e = self.e

        return 'quad'+str(s[0])+str(s[1])+str(c[0])+str(c[1])+str(e[0])+str(e[1])

    def plotSpline(self): #doesn't show() it on purpose, to be used later by parent function call       
        s = self.s
        c = self.c
        e = self.e

        params = array(range(101))/100.0
        sp_pts = []
        for t in params: #based on the formula for quaratic bezier splines
            x_t = s[0]*pow(1-t,2) + c[0]*(2*t)*(1-t) + e[0]*pow(t,2)
            y_t = s[1]*pow(1-t,2) + c[1]*(2*t)*(1-t) + e[1]*pow(t,2)
            sp_pts.append([x_t,y_t])

        sp_pts = array(sp_pts)
        plt.plot(sp_pts[:,0], sp_pts[:,1], 'g-', linewidth=3)


class CubBezSpline:
    def __init__(self, start, ctl1, ctl2, end):
        self.s = start
        self.c1 = ctl1
        self.c2 = ctl2
        self.e = end

    def closestXGivenY(self, vts, vts_h):
        # cubic function a*t^3 + b*t^2 + c*t + d = 0, using vt_iy
        # solve for t given a,b,c,d, use it to find closest x to the right of vt
        # the coeffs are derived from cubic bez formula

        s = array([self.s]*len(vts)).astype(complex)
        c1 = array([self.c1]*len(vts)).astype(complex)
        c2 = array([self.c2]*len(vts)).astype(complex)
        e = array([self.e]*len(vts)).astype(complex)
        
        a = e[:,1] - 3*c2[:,1] + 3*c1[:,1] - s[:,1]
        b = 3*c2[:,1] - 6*c1[:,1] + 3*s[:,1]
        c = 3*c1[:,1] - 3*s[:,1]
        d = s[:,1] - vts[:,1]

        p = -b/(3*a)
        q = p**3 + (b*c - 3*a*d)/(6*a**2)
        r = c/(3*a)
        x1 = (q + (q**2 + (r-p*p)**3)**.5)**(1.0/3)
        x2 = (q - (q**2 + (r-p*p)**3)**.5)**(1.0/3)
        x3 = p
        t1 = x1+x2+x3

        m = array([a, b, c, d]).T
        a_n = m[:,0]
        b_n = m[:,1] + a_n*t1
        c_n = m[:,2] + b_n*t1
        t2 = (-1*b_n + (b_n*b_n - 4*a_n*c_n)**.5)/(2*a_n)
        t3 = (-1*b_n - (b_n*b_n - 4*a_n*c_n)**.5)/(2*a_n)

        ts = [t1, t2, t3]
        closest_xs = array([Inf]*len(vts)).astype(complex)
        closest_xsh = array(['.']*len(vts)).astype(str_)
        for t in ts:
            isValid = (.00001<=t)*(t<=1.00001)
            x_ts = ((1-t)**3)*s[:,0] + 3*((t-1)**2)*t*c1[:,0] + 3*(1-t)*t*t*c2[:,0] + (t**3)*e[:,0]
            
            spline_h = self.getSplineHash()
            x_tsh = array([i+spline_h for i in x_ts.astype(str_)]).astype(str_) #calculated points' hash array
            
            isNotSame = x_tsh!=vts_h
            isClosest = (x_ts < closest_xs)*(x_ts-vts[:,0]>0)*isValid*isNotSame
            closest_xs[isClosest] = x_ts[isClosest]
            closest_xsh[isClosest] = x_tsh[isClosest]

        closest_xs[closest_xs==Inf] = array(['.']*sum(closest_xs==Inf)).astype(str_)

        '''
        closest_x = '.'
        for t in ts:

            if (arg(t) == 0 or math.isnan(arg(t))) and (t >= 0 and t <= 1): #checks to make sure real root and is valid param val
                x_t = s[0]*pow(1-t,3) + c1[0]*3*pow(1-t,2)*t + c2[0]*3*(1-t)*pow(t,2) + e[0]*pow(t,3)

                x_th = str(x_t)+self.getSplineHash()

                if x_t - vt_i[0] > 0 and closest_x == '.' and not x_th == vt_ih:
                    closest_x = x_t
                elif x_t - vt_i[0] > 0 and x_t < closest_x and not x_th == vt_ih: #ensures closest pt isn't vt_ix
                    closest_x = x_t
        '''
        return [closest_xs, closest_xsh]

    def smallestX(self):
        return min(self.s[0], self.c1[0], self.c2[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.c1[0], self.c2[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.c1[1], self.c2[1], self.e[1])

    def biggestY(self):
        return max(self.s[1], self.c1[1], self.c2[1], self.e[1])

    def getSplineHash(self):
        s = self.s
        c1 = self.c1
        c2 = self.c2
        e = self.e

        return 'cub'+str(s[0])+str(s[1])+str(c1[0])+str(c1[1])+str(c2[0])+str(c2[1])+str(e[0])+str(e[1])

    def plotSpline(self): #doesn't show() it on purpose, to be used later by parent function call                                                                                                         
        s = self.s
        c1 = self.c1
        c2 = self.c2
        e = self.e

        params = array(range(101))/100.0
        sp_pts = []
        for t in params: #based on the formula for cubic bezier splines
            x_t = s[0]*pow(1-t,3) + c1[0]*3*pow(1-t,2)*t + c2[0]*3*(1-t)*pow(t,2) + e[0]*pow(t,3)
            y_t = s[1]*pow(1-t,3) + c1[1]*3*pow(1-t,2)*t + c2[1]*3*(1-t)*pow(t,2) + e[1]*pow(t,3)
            sp_pts.append([x_t,y_t])

        sp_pts = array(sp_pts)
        plt.plot(sp_pts[:,0], sp_pts[:,1], 'g-', linewidth=3)

class ArcSpline:
    def __init__(self, start, radx, rady, x_rotation, large_arc_flag, sweep_flag, end):
        self.s = start
        self.rx = radx
        self.ry = rady
        self.xrot = x_rotation
        self.laf = large_arc_flag
        self.sf = sweep_flag
        self.e = end

    def closestXGivenY(self, vt_i, vt_ih):
        s = self.s
        rx = self.rx
        ry = self.ry
        xrot = self.xrot
        laf = self.laf
        sf = self.sf
        e = self.e

        x = symbol('x')
        y = symbol('y')
