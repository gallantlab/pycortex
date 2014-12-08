from pylab import *
import math
import time

class LineSpline:
    def __init__(self, start, end):
        self.s = zeros(2)
        self.e = zeros(2)

        self.s[0] = start[0]
        self.s[1] = start[1]
        self.e[0] = end[0]
        self.e[1] = end[1]

    def allSplineXGivenY(self, vts):
        # linear function t*s + (1-t)*e = vt_i
        # solve for t given y-coords, find closest x-coord

        s = self.s
        e = self.e

        isHorizLine = s[1]==e[1]

        a = e[1] - s[1]
        b = s[1] - vts[:,1]
        
        # params for the single root
        t = nan
        if isHorizLine:
            t = -1*ones(vts.shape[0])
            isSSmaller = s[0]<e[0]
            isAtY = vts[:,1]==s[1]
            t[isAtY] = t[isAtY] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<e, else t=1
        else:
            t = -1*b/a

        closest_xs = Inf*ones(vts.shape[0])

        # ensure it's within the spline's piecewise region   
        isValid = (0<=t)*(t<=1)
        x_ts = (1-t)*s[0] + t*e[0]
        
        isClosest = (x_ts-vts[:,0]>0)*isValid

        closest_xs[isClosest] = x_ts[isClosest]

        return closest_xs


    def allSplineYGivenX(self, vts):
        # linear function t*s + (1-t)*e = vt_i
        # solve for t given x-coords, find closest y-coord

        s = self.s
        e = self.e

        isVertLine = s[0]==e[:,0]

        a = e[0] - s[:,0]
        b = s[0] - vts[:,0]

        # params for the single root  
        t = nan
        if isVertLine:
            t = -1*ones(vts.shape[0])
            isSSmaller = s[1]<e[1]
            isAtX = vts[:,0]==s[0]
            t[isAtX] = t[isAtX] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<e, else t=1
        else:
            t = -1*b/a

        closest_ys = Inf*ones(vts)

        # ensure it's within the spline's piecewise region
        isValid = (0<=t)*(t<=1)
        y_ts = (1-t)*s[1] + t*e[1]
        
        isClosest = (y_ts-vts[:,1]>0)*isValid
        closest_ys[isClosest] = y_ts[isClosest]
        return closest_ys

    def smallestX(self):
        return min(self.s[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.e[1])

    def biggestY(self):
         return max(self.s[1], self.e[1])

    def toString(self):
        s = self.s
        e = self.e

        return 'lin'+str(s[0])+str(s[1])+str(e[0])+str(e[1])
    
    def translateSpline(self, ref):
        self.s += ref
        self.e += ref
    
    def plotSpline(self, ind = -1):
        s = self.s
        e = self.e

        params = array(range(101))/100.0
        sp_pts = []
        for t in params: #based on the formula for linear splines
            x_t = float(t*s[0] + (1-t)*e[0])
            y_t = float(t*s[1] + (1-t)*e[1])
            sp_pts.append([x_t,y_t])

        sp_pts = array(sp_pts)
        if ind == 0:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'm.', linewidth=3)
        elif ind == 1:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'y.', linewidth=3)
        else:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'g.', linewidth=3)

class QuadBezSpline:
    def __init__(self, start, ctl, end):
        self.s = zeros(2)
        self.c = zeros(2)
        self.e = zeros(2) 

        self.s[0] = start[0]
        self.s[1] = start[1]
        self.c[0] = ctl[0]
        self.c[1] = ctl[1]
        self.e[0] = end[0]
        self.e[1] = end[1]

    def allSplineXGivenY(self, vts):
        # quadtric function a*t^2 + b*t + c = 0, using vt_iy
        # solve for t given a,b,c, use it to find closest x to the right of vt
        # coeffs are derived from the quad bez formua

        s = self.s
        c = self.c
        e = self.e

        a = s[1] - 2*c[1] + e[1]
        b = 2*c[1] - 2*s[1]
        c = s[1] - vts[:,1]

        if a == 0:
            t = nan
            if b == 0:
                t = -1*ones(vts.shape[0])
                isSSmaller = s[0]<c[0]
                isAtY = vts[:,1]==s[1]
                t[isAtY] = t[isAtY] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<c, else t=1                                                                                                                                              \
                    
            else:
                t = -1.0*c/b
                
            closest_xs = Inf*ones((vts.shape[0],2))
            #ensure it's within the spline's piecewise region

            isValid = (0<=t)*(t<=1)
            x_ts = ((1-t)**2)*s[0] + 2*(1-t)*t*c[0] + t*t*e[0]
            isClosest = (x_ts-vts[:,0]>0)*isValid
            for i in range(closest_xs.shape[1]):
                closest_xs[isClosest,i] = x_ts[isClosest]
            return closest_xs

        # params for the two roots
        t1 = (-b + (b*b - 4*a*c)**.5)/(2*a) 
        t2 = (-b - (b*b - 4*a*c)**.5)/(2*a)

        closest_xs = Inf*ones((vts.shape[0],2))

        # ensure they're within the spline's piecewise region
        isValid1 = (0<=t1)*(t1<=1)
        isValid2 = (0<=t2)*(t2<=1)

        x_t1 = ((1-t1)**2)*s[0] + 2*(1-t1)*t1*c[0] + t1*t1*e[0]
        x_t2 = ((1-t2)**2)*s[0] + 2*(1-t2)*t2*c[0] + t2*t2*e[0]

        closest_xs[isValid1,0] = x_t1[isValid1]
        closest_xs[isValid2,1] = x_t2[isValid2]

        return closest_xs

    def allSplineYGivenX(self, vts):
        # quadtric function a*t^2 + b*t + c = 0, using vt_iy
        # solve for t given a,b,c, use it to find closest y above vt
        # coeffs are derived from the quad bez formua                                                                 

        s = self.s
        c = self.c
        e = self.e

        a = s[0] - 2*c[0] + e[0]
        b = 2*c[0] - 2*s[0]
        c = s[0] - vts[:,0]

        if a == 0:
            t = nan
            if b == 0:
                t = -1*ones(vts.shape[0])
                isSSmaller = s[1]<c[1]
                isAtX = vts[:,0]==s[0]
                t[isAtX] = t[isAtX] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<c, else t=1
            else:
                t = -1.0*c/b
                
            closest_ys = Inf*ones((vts.shape[0],2))
            #ensure it's within the spline's piecewise region 

            isValid = (0<=t)*(t<=1)
            y_ts = ((1-t)**2)*s[1] + 2*(1-t)*t*c[1] + t*t*e[1]
            isClosest = (x_ts-vts[:,1]>0)*isValid
            for i in range(closest_xs.shape[1]):
                closest_xs[isClosest,i] = x_ts[isClosest]
            return closest_xs

        # params for the two roots  
        t1 = (-b + (b*b - 4*a*c)**.5)/(2*a)
        t2 = (-b - (b*b - 4*a*c)**.5)/(2*a)

        closest_ys = Inf*ones((vts.shape[0],2))

        # ensure they're within the spline's piecewise region   
        isValid1 = (0<=t1)*(t1<=1)
        isValid2 = (0<=t2)*(t2<=1)

        y_t1 = ((1-t1)**2)*s[1] + 2*(1-t1)*t1*c[1] + t1*t1*e[1]
        y_t2 = ((1-t2)**2)*s[1] + 2*(1-t2)*t2*c[1] + t2*t2*e[1]

        closest_ys[isValid1,0] = y_t1[isValid1]
        closest_ys[isValid2,1] = y_t2[isValid2]

        return closest_ys

    def smallestX(self):
        return min(self.s[0], self.c[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.c[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.c[1], self.e[1])

    def biggestY(self):
        return max(self.s[1], self.c[1], self.e[1])

    def translateSpline(self, ref):
        self.s += ref
        self.c += ref
        self.e += ref

    def toString(self):
        s = self.s
        c = self.c
        e = self.e

        return 'quad'+str(s[0])+str(s[1])+str(c[0])+str(c[1])+str(e[0])+str(e[1])

    def plotSpline(self, ind = -1): #doesn't show() it on purpose, to be used later by parent function call       
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
        if ind == 0:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'm.', linewidth=3)
        elif ind == 1:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'y.', linewidth=3)
        else:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'b.', linewidth=3)


class CubBezSpline:
    def __init__(self, start, ctl1, ctl2, end):
        self.s = zeros(2)
        self.c1 = zeros(2)
        self.c2 = zeros(2)
        self.e = zeros(2)

        self.s[0] = start[0]
        self.s[1] = start[1]
        self.c1[0] = ctl1[0]
        self.c1[1] = ctl1[1]
        self.c2[0] = ctl2[0]
        self.c2[1] = ctl2[1]
        self.e[0] = end[0]
        self.e[1] = end[1]

    def allSplineXGivenY(self, vts):
        # cubic function a*t^3 + b*t^2 + c*t + d = 0, using vt_iy
        # solve for t given a,b,c,d, use it to find closest x to the right of vt
        # the coeffs are derived from cubic bez formula
        # based on computationally safe precision variant of Cardano's method from this link:
        # https://www.e-education.psu.edu/png520/m11_p6.html

        s = self.s
        c1 = self.c1
        c2 = self.c2
        e = self.e

        a = e[1] - 3.0*c2[1] + 3.0*c1[1] - s[1]
        b = 3.0*c2[1] - 6.0*c1[1] + 3.0*s[1]
        c = 3.0*c1[1] - 3.0*s[1]
        d = s[1] - vts[:,1]

        if a == 0:
            if b == 0:
                t = nan
                if c == 0:
                    t = -1*ones(vts.shape[0])
                    isSSmaller = s[0]<c1[0]
                    isAtY = vts[:,1]==s[1]
                    t[isAtY] = t[isAtY] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<e, else t=1                                                                                                                                                       
                else:
                    t = -1.0*d/c

                closest_xs = Inf*ones((vts.shape[0],3))
                #ensure it's within the spline's piecewise region 
                isValid = (0<=t)*(t<=1)
                x_ts = ((1-t)**3)*s[0] + 3*((1-t)**2)*t*c1[0] + 3*(1-t)*t*t*c2[0] + (t**3)*e[0]
                isClosest = (x_ts-vts[:,0]>0)*isValid
                for i in range(closest_xs.shape[1]):
                    closest_xs[isClosest,i] = x_ts[isClosest]                
                return closest_xs
            else:
                # params for the two roots
                t1 = (-c + (b*b - 4*b*d)**.5)/(2*b)
                t2 = (-c - (b*b - 4*b*d)**.5)/(2*b)

                closest_xs = Inf*ones((vts.shape[0],3))
        
                #ensure they're within the spline's piecewise region
                isValid1 = (0<=t1)*(t1<=1)
                isValid2 = (0<=t2)*(t2<=1)

                x_t1 = ((1-t1)**3)*s[0] + 3*((1-t1)**2)*t1*c1[0] + 3*(1-t1)*t1*t1*c2[0] + (t1**3)*e[0]
                x_t2 = ((1-t2)**3)*s[0] + 3*((1-t2)**2)*t2*c1[0] + 3*(1-t2)*t2*t2*c2[0] + (t2**3)*e[0]

                closest_xs[isValid1,0] = x_t1[isValid1]
                closest_xs[isValid2,1] = x_t2[isValid2]
                closest_xs[isValid2,2] = x_t2[isValid2]
                
                return closest_xs
    
        #standardize to x^3 + a*x^2 + b*x + c = 0 format
        a_old = a
        a = b/a_old
        b = c/a_old
        c = d/a_old

        Q = (a**2 - 3.0*b)/9.0
        R = (2.0*a**3 - 9.0*a*b + 27.0*c)/54.0
        M = R**2 - Q**3
        
        isMPos = M>0
        isMPos = array([isMPos]).T
        
        #if M<0, 3 real roots
        theta = arccos(R/((Q**3)**.5))

        x1 = real(-1.0*(2.0*(Q**.5)*cos(theta/3.0)) - a/3.0)
        x2 = real(-1.0*(2.0*(Q**.5)*cos((theta + 2.0*pi)/3.0)) - a/3.0)
        x3 = real(-1.0*(2.0*(Q**.5)*cos((theta - 2.0*pi)/3.0)) - a/3.0)

        #if M>0, 1 real root
        S_p = -1.0*sign(R)*(abs(R) + M**.5)**(1.0/3)
        T_p = Q/S_p
        x_p = S_p + T_p - a/3.0

        ts1 = ~isMPos*array([x1, x2, x3]).T
        isNaNts1 = isnan(ts1)
        ts1[isNaNts1] = 0.0

        ts2 = isMPos*array([x_p,x_p,x_p]).T
        isNaNts2 = isnan(ts2)
        ts2[isNaNts2] = 0.0

        ts = ts1 + ts2
        ts[isNaNts1*isNaNts2] = -1.0

        # params for the three roots
        t1 = ts[:,0]
        t2 = ts[:,1]
        t3 = ts[:,2]

        closest_xs = Inf*ones((vts.shape[0],3))

        # ensure they're within the spline's piecewise region   
        isValid1 = (0<=t1)*(t1<=1)
        isValid2 = (0<=t2)*(t2<=1)
        isValid3 = (0<=t3)*(t3<=1)

        x_t1 = ((1-t1)**3)*s[0] + 3*((1-t1)**2)*t1*c1[0] + 3*(1-t1)*t1*t1*c2[0] + (t1**3)*e[0]
        x_t2 = ((1-t2)**3)*s[0] + 3*((1-t2)**2)*t2*c1[0] + 3*(1-t2)*t2*t2*c2[0] + (t2**3)*e[0]
        x_t3 = ((1-t3)**3)*s[0] + 3*((1-t3)**2)*t3*c1[0] + 3*(1-t3)*t3*t3*c2[0] + (t3**3)*e[0]

        closest_xs[isValid1,0] = x_t1[isValid1]
        closest_xs[isValid2,1] = x_t2[isValid2]
        closest_xs[isValid3,2] = x_t3[isValid3]

        return closest_xs

    def allSplineYGivenX(self, vts):
        # cubic function a*t^3 + b*t^2 + c*t + d = 0, using vt_ix 
        # solve for t given a,b,c,d, use it to find closest y to the right of vt
        # the coeffs are derived from cubic bez formula
        # based on computationally safe precision variant of Cardano's method from this link:
        # https://www.e-education.psu.edu/png520/m11_p6.html

        s = self.s
        c1 = self.c1
        c2 = self.c2
        e = self.e

        a = e[0] - 3.0*c2[0] + 3.0*c1[0] - s[0]
        b = 3.0*c2[0] - 6.0*c1[0] + 3.0*s[0]
        c = 3.0*c1[0] - 3.0*s[0]
        d = s[0] - vts[:,0]

        if a == 0:
            if b == 0:
                t = nan
                if c == 0:
                    t = -1*ones(vts.shape[0])
                    isSSmaller = s[1]<c1[1]
                    isAtX = vts[:,0]==s[0]
                    t[isAtX] = t[isAtX] + 1*isSSmaller + 2*(~isSSmaller) # t=0 if s<c1, else t=1                                                                                                                                                                                                                                              
                else:
                    t = -1.0*d/c

                closest_ys = Inf*ones((vts.shape[0],3))
                #ensure it's within the spline's piecewise region                                
                isValid = (0<=t)*(t<=1)
                y_ts = ((1-t)**3)*s[1] + 3*((1-t)**2)*t*c1[1] + 3*(1-t)*t*t*c2[1] + (t**3)*e[1]
                isClosest = (y_ts-vts[:,1]>0)*isValid
                for i in range(closest_ys.shape[1]):
                    closest_ys[isClosest,i] = y_ts[isClosest]
                return closest_ys
            else:
                # params for the two roots                                            
                t1 = (-c + (b*b - 4*b*d)**.5)/(2*b)
                t2 = (-c - (b*b - 4*b*d)**.5)/(2*b)

                closest_ys = Inf*ones((vts.shape[0],3))

                #ensure they're within the spline's piecewise region
                isValid1 = (0<=t1)*(t1<=1)
                isValid2 = (0<=t2)*(t2<=1)

                y_t1 = ((1-t1)**3)*s[1] + 3*((1-t1)**2)*t1*c1[1] + 3*(1-t1)*t1*t1*c2[1] + (t1**3)*e[1]
                y_t2 = ((1-t2)**3)*s[1] + 3*((1-t2)**2)*t2*c1[1] + 3*(1-t2)*t2*t2*c2[1] + (t2**3)*e[1]

                closest_ys[isValid1,0] = x_t1[isValid1]
                closest_ys[isValid2,1] = x_t2[isValid2]
                closest_ys[isValid2,2] = x_t2[isValid2]

                return closest_ys
        
        #standardize to y^3 + a*y^2 + b*y + c = 0 format   
        a_old = a
        a = b/a_old
        b = c/a_old
        c = d/a_old

        Q = (a**2 - 3.0*b)/9.0
        R = (2.0*a**3 - 9.0*a*b + 27*c)/54.0
        M = R**2 - Q**3
        isMPos = M>0
        isMPos = array([isMPos]).T

        #if M<0, 3 real roots
        theta = arccos(R/((Q**3)**.5))
        y1 = real(-1.0*(2.0*(Q**.5)*cos(theta/3.0)) - a/3.0)
        y2 = real(-1.0*(2.0*(Q**.5)*cos((theta + 2.0*pi)/3.0)) - a/3.0)
        y3 = real(-1.0*(2.0*(Q**.5)*cos((theta - 2.0*pi)/3.0)) - a/3.0)

        #if M>0, 1 real root
        S_p = -1.0*sign(R)*(abs(R) + M**.5)**(1.0/3)
        T_p = Q/S_p
        y_p = S_p + T_p - a/3.0
        y_p = y_p

        ts1 = ~isMPos*array([y1, y2, y3]).T
        isNaNts1 = isnan(ts1)
        ts1[isNaNts1] = 0
        ts2 = isMPos*array([y_p,y_p,y_p]).T
        isNaNts2 = isnan(ts2)
        ts2[isNaNts2] = 0

        ts = ts1 + ts2
        ts[isNaNts1*isNaNts2] = -1.0

        # params for the three roots
        t1 = ts[:,0]
        t2 = ts[:,1]
        t3 = ts[:,2]

        closest_ys = Inf*ones((vts.shape[0],3))

        # ensure they're within the spline's piecewise region   
        isValid1 = (0<=t1)*(t1<=1)
        isValid2 = (0<=t2)*(t2<=1)
        isValid3 = (0<=t3)*(t3<=1)

        y_t1 = ((1-t1)**3)*s[1] + 3*((1-t1)**2)*t1*c1[1] + 3*(1-t1)*t1*t1*c2[1] + (t1**3)*e[1]
        y_t2 = ((1-t2)**3)*s[1] + 3*((1-t2)**2)*t2*c1[1] + 3*(1-t2)*t2*t2*c2[1] + (t2**3)*e[1]
        y_t3 = ((1-t3)**3)*s[1] + 3*((1-t3)**2)*t3*c1[1] + 3*(1-t3)*t3*t3*c2[1] + (t3**3)*e[1]

        closest_ys[isValid1,0] = y_t1[isValid1]
        closest_ys[isValid2,1] = y_t2[isValid2]
        closest_ys[isValid3,2] = y_t3[isValid3]

        return closest_ys

    def smallestX(self):
        return min(self.s[0], self.c1[0], self.c2[0], self.e[0])

    def biggestX(self):
        return max(self.s[0], self.c1[0], self.c2[0], self.e[0])

    def smallestY(self):
        return min(self.s[1], self.c1[1], self.c2[1], self.e[1])

    def biggestY(self):
        return max(self.s[1], self.c1[1], self.c2[1], self.e[1])
    
    def translateSpline(self, ref):
        self.s += ref
        self.c1 += ref
        self.c2 += ref
        self.e += ref

    def toString(self):
        s = self.s
        c1 = self.c1
        c2 = self.c2
        e = self.e

        return 'cub'+str(s[0])+str(s[1])+str(c1[0])+str(c1[1])+str(c2[0])+str(c2[1])+str(e[0])+str(e[1])

    def plotSpline(self, ind = -1): #doesn't show() it on purpose, to be used later by parent function call
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
        if ind == 0:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'm.', linewidth=3)
        elif ind == 1:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'y.', linewidth=3)
        else:
            plt.plot(sp_pts[:,0], sp_pts[:,1], 'k.', linewidth=3)


class ArcSpline: ### INCOMPLETE. DO NOT USE.
    def __init__(self, start, radx, rady, x_rotation, large_arc_flag, sweep_flag, end):
        self.s = start
        self.rx = radx
        self.ry = rady
        self.xrot = x_rotation
        self.laf = large_arc_flag
        self.sf = sweep_flag
        self.e = end

    def allSplineXGivenY(self, vt_i, vt_ih):
        s = self.s
        rx = self.rx
        ry = self.ry
        xrot = self.xrot
        laf = self.laf
        sf = self.sf
        e = self.e

        x = symbol('x')
        y = symbol('y')
