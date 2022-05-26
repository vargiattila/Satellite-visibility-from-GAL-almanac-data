import math
import numpy as np
import pandas as pd



def satpos(gps_week, sec_of_week, prn, health, ecc, ax, raw, aop, man, toa, inc, rra, week):
    xsat = []
    Wedot = 7.2921151467e-5;	#WGS 84 value of earth's rotation rate
    mu =  3.986005e+14;		#WGS 84 value of earth's univ. grav. par.
    #mean motion
    n = math.sqrt(mu/float(ax)**6);
    #print(prn,n)
    T = 2.0 * math.pi / n;
    #print(T)
    dt = sec_of_week - float(toa);
    #print(dt)
    if abs(dt) > 604800:
        print('*** to much time difference %f\n', dt);
    else:
        M = float(man) + n * dt;
        while (M<0):
            M = M + np.pi
        #print(M)
    #Kepler equation
        E = M;
        Eold = 0.0;
        j = 0;
        while (abs(E - Eold) > 1.0e-8):
            Eold = E
            E = M + float(ecc) * math.sin(E)
            j += 1
            #print(prn,E)
    #true anomaly
        snu = math.sqrt(1.0-float(ecc)**2)*math.sin(E)
        cnu = math.cos(E)-float(ecc)
        nu = math.atan2(snu, cnu)
        #print(nu)
    #position in orbit plane
        u = nu+float(aop)
        r = float(ax**2)*(1.0-float(ecc)*math.cos(E))
        #print(r)
        wc = float(raw)+(float(rra)-Wedot)*dt-(float(toa)*Wedot) #felszálló csomópont
        #print(wc)
        xdash = r*math.cos(nu)
        ydash = r*math.sin(nu)
        zdash = 0
        xsatorb = np.array([[xdash],[ydash],[zdash]])
        #print(xsatorb)
    #position in ECEF system
        Rz = np.array([[np.cos(aop), -np.sin(aop),0],\
                      [np.sin(aop), np.cos(aop),0],\
                      [0,0,1]]);
        Rx = np.array([[1,0,0],\
                      [0, np.cos(inc),-np.sin(inc)],\
                      [0,np.sin(inc),np.cos(inc)]]);
        Rzo = np.array([[np.cos(wc), -np.sin(wc),0],\
                      [np.sin(wc), np.cos(wc),0],\
                      [0,0,1]]);
        xsat = Rzo.dot(Rx.dot(Rz.dot(xsatorb)))
    return xsat

def ymdhms2gps(year, month, mday, hour, minute, second):
    regu_month_day= [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leap_month_day= [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    if year % 4 == 0:
        yday = leap_month_day[month-1] + mday
    else:
        yday = regu_month_day[month-1] + mday
    mjd = math.trunc(((year - 1901) / 4)) * 1461 + math.trunc(((year - 1901) % 4)) * 365 + yday - 1 + 15385
    fmjd = ((second / 60.0 + minute) / 60.0 + hour) / 24.0
    gps_week = math.trunc((mjd - 44244) / 7)
    sec_of_week = ((mjd - 44244) - gps_week * 7 + fmjd) * 86400
    return gps_week,sec_of_week


ecc = 0.011672973630
ax = 5153.441895
raw = -0.116940618
aop = 0.917901397
man = -0.310368896
toa = 405504
inc = 0.8884773254
rra = -8.414644981*10e-10
week = 605
prn = 11
health = 1

[gps_week, sec_of_week] = ymdhms2gps(2011,3,31,8,14,59)

a = satpos(gps_week, sec_of_week, prn, health, ecc, ax, raw, aop, man, toa, inc, rra, week)

print(a)
