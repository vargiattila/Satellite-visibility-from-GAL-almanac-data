import math
import re
import matplotlib.pyplot as plt
import pandas as pd
import sys
import gnsscal
import datetime
import numpy as np
from datetime import datetime
from itertools import takewhile, islice, dropwhile
from numpy import linalg as LA
import matplotlib.animation as animation
from celluloid import Camera
pd.set_option('display.max_columns', None)


def read_xml(xml_file):
    data = []
    ID = []
    aSqRoot = []
    ecc = []
    deltai = []
    omega0 = []
    omegaDot = []
    w = []
    m0 = []
    af0 = []
    af1 = []
    iod = []
    t0a = []
    wna = []
    statusE5a = []
    statusE5b = []
    statusE1B = []
    
    almanac = open(xml_file, 'r')
    
    for line in almanac:
        line = line.strip()
        data.append(line)
    
    for line in data:
        if re.findall('<SVID>', line):
            result = re.search('<SVID>(.*)</SVID>', line)
            ID.append(result.group(1))
        elif re.match('<aSqRoot>', line):
            result = re.search('<aSqRoot>(.*)</aSqRoot>', line)
            aSqRoot.append(result.group(1))
        elif re.match('<ecc>', line):
            result = re.search('<ecc>(.*)</ecc>', line)
            ecc.append(result.group(1))
        elif re.match('<deltai>', line):
            result = re.search('<deltai>(.*)</deltai>', line)
            deltai.append(result.group(1))
        elif re.match('<omega0>', line):
            result = re.search('<omega0>(.*)</omega0>', line)
            omega0.append(result.group(1))
        elif re.match('<omegaDot>', line):
            result = re.search('<omegaDot>(.*)</omegaDot>', line)
            omegaDot.append(result.group(1))
        elif re.match('<w>', line):
            result = re.search('<w>(.*)</w>', line)
            w.append(result.group(1))
        elif re.match('<m0>', line):
            result = re.search('<m0>(.*)</m0>', line)
            m0.append(result.group(1))
        elif re.match('<af0>', line):
            result = re.search('<af0>(.*)</af0>', line)
            af0.append(result.group(1))
        elif re.match('<af1>', line):
            result = re.search('<af1>(.*)</af1>', line)
            af1.append(result.group(1))
        elif re.match('<iod>', line):
            result = re.search('<iod>(.*)</iod>', line)
            iod.append(result.group(1))
        elif re.match('<t0a>', line):
            result = re.search('<t0a>(.*)</t0a>', line)
            t0a.append(result.group(1))
        elif re.match('<wna>', line):
            result = re.search('<wna>(.*)</wna>', line)
            wna.append(result.group(1))
        elif re.match('<statusE5a>', line):
            result = re.search('<statusE5a>(.*)</statusE5a>', line)
            statusE5a.append(result.group(1))
        elif re.match('<statusE5b>', line):
            result = re.search('<statusE5b>(.*)</statusE5b>', line)
            statusE5b.append(result.group(1))
        elif re.match('<statusE1B>', line):
            result = re.search('<statusE1B>(.*)</statusE1B>', line)
            statusE1B.append(result.group(1))
            
    columns = ['ID','aSqRoot','ecc','deltai','omega0','omegadot','w',\
            'm0','af0','af1','iod','t0a','wna','statusE5a','statusE5b','statusE1B']
    
    sat_data = pd.DataFrame(columns=columns)
    sat_data['ID'] = ID
    sat_data['aSqRoot'] = aSqRoot
    sat_data['ecc'] = ecc
    sat_data['deltai'] = deltai
    sat_data['omega0'] = omega0
    sat_data['omegadot'] = omegaDot
    sat_data['w'] = w
    sat_data['m0'] = m0
    sat_data['af0'] = af0
    sat_data['af1'] = af1
    sat_data['iod'] = iod
    sat_data['t0a'] = t0a
    sat_data['wna'] = wna
    sat_data['statusE5a'] = statusE5a
    sat_data['statusE5b'] = statusE5b
    sat_data['statusE1B'] = statusE1B
    sat_data = sat_data.astype(float)
    sat_data['ID'] = sat_data['ID'].astype(int)
    sat_data['deltai'] = np.radians(56)+sat_data['deltai']*np.pi
    sat_data['omega0'] = sat_data['omega0']*np.pi
    sat_data['omegadot'] = sat_data['omegadot']*np.pi
    sat_data['w'] = sat_data['w']*np.pi
    sat_data['m0'] = sat_data['m0']*np.pi
    sat_data['ax'] = (math.sqrt(29600000) + sat_data['aSqRoot'])**2
    return sat_data 

def lla2ecef(lat, lon, alt):
    rad_lat = np.radians(lat)
    rad_lon = np.radians(lon)
    a = 6378137
    e = 8.1819190842622e-2

    N = a / np.sqrt(1 - e ** 2 * np.sin(rad_lat) ** 2)

    x = (N + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (N + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = ((1 - e ** 2) * N + alt) * np.sin(rad_lat)

    return x, y, z


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
    sec_of_week = ((mjd - 44244) - gps_week * 7 + fmjd) * 86400+18
    return gps_week,sec_of_week

def satpos(gps_week, sec_of_week, prn, health, ecc, ax, raw, aop, man, toa, inc, rra, week):
    xsat = []
    Wedot = 7.2921151467e-5;	#WGS 84 value of earth's rotation rate
    mu =  3.986005e+14;		#WGS 84 value of earth's univ. grav. par.
    #mean motion
    n = math.sqrt(mu/float(ax)**3);
    T = 2.0 * math.pi / n;
    dt = sec_of_week - float(toa);
    if abs(dt) > 604800:
        print('*** to much time difference %f\n', dt);
    else:
        M = float(man) + n * dt;
        while (M<0):
            M = M + np.pi
    #Kepler equation
        E = M;
        Eold = 0.0;
        j = 0;
        while (abs(E - Eold) > 1.0e-8):
            Eold = E
            E = M + float(ecc) * math.sin(E)
            j += 1
    #true anomaly
        snu = math.sqrt(1.0-float(ecc)**2)*math.sin(E)
        cnu = math.cos(E)-float(ecc)
        nu = math.atan2(snu, cnu)
        #print('nu=',nu)
    #position in orbit plane
        u = nu+float(aop)
        r = float(ax)*(1.0-float(ecc)*math.cos(E))
        wc = float(raw)+(float(rra)-Wedot)*dt-(float(toa)*Wedot) #felsz??ll?? csom??pont
        xdash = r*math.cos(u)
        ydash = r*math.sin(u)
        zdash = 0
        xsatorb = np.array([[xdash],[ydash],[zdash]])
    #position in ECEF system
        xsat.append(float(xdash)*math.cos(float(wc)) - float(ydash)*math.cos(float(inc))*math.sin(float(wc)))
        xsat.append(float(xdash)*math.sin(float(wc)) + float(ydash)*math.cos(float(inc))*math.cos(float(wc)))
        xsat.append(float(ydash)*math.sin(float(inc)))
    return xsat

def ECEF2GPS(Pos):
    x=Pos[0];
    y=Pos[1];
    z=Pos[2];
    ##WGS84 ellipsoid constants:
    a = 6378137;
    e = 8.1819190842622e-2;
    ##calculations:
    b   = np.sqrt(a**2*(1-e**2));
    ep  = np.sqrt((a**2-b**2)/b**2);
    p   = np.sqrt(x**2+y**2);
    th  = np.arctan2(a*z,b*p);
    lon = np.arctan2(y,x);
    lond = np.degrees(np.arctan2(y,x));
    lat = np.arctan((z+ep**2*b*(math.sin(th))**3)/(p-e**2*a*(math.cos(th))**3));
    latd = np.degrees(np.arctan((z+ep**2*b*(math.sin(th))**3)/(p-e**2*a*(math.cos(th))**3)));
    N   = a/np.sqrt(1-e**2*(math.sin(lat))**2);
    alt = p/math.cos(lat)-N;
    GPS=[lat,lon,alt];
    return GPS

def GPS2ECEF(lat,lon,alt):
    ##WGS84 ellipsoid constants:
    a = 6378137;
    e = 8.1819190842622e-2;
    ##intermediate calculation
    ##(prime vertical radius of curvature)
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2);
    ##results:
    x = (N+alt) * np.cos(lat) * np.cos(lon);
    y = (N+alt) * np.cos(lat) * np.sin(lon);
    z = ((1-e**2) * N + alt) * np.sin(lat);
    xyz = [x,y,z];
    return xyz

def XYZ2ENU(A,Phi,Lambda): 
  XYZ2ENU= np.array([[-np.sin(Lambda), np.cos(Lambda), 0],\
           [-np.sin(Phi)*np.cos(Lambda), -np.sin(Phi)*np.sin(Lambda), np.cos(Phi)],\
           [np.cos(Phi)*np.cos(Lambda), np.cos(Phi)*np.sin(Lambda),  np.sin(Phi)]])
  A = np.array(A)
  ENU=XYZ2ENU.dot(A)
  return ENU

def Calc_Azimuth_Elevation(Pos_Rcv,Pos_SV):
    Pos_Rcv = np.array(Pos_Rcv)
    Pos_SV = np.array(Pos_SV)
    R=Pos_SV-Pos_Rcv;               ##vector from Reciever to Satellite
    GPS = ECEF2GPS(Pos_Rcv);        ##Lattitude and Longitude of Reciever
    Lat=GPS[0];Lon=GPS[1];
    ENU=XYZ2ENU(R,Lat,Lon);
    Elevation=np.arcsin(ENU[2]/LA.norm(ENU));
    Azimuth=np.arctan2(ENU[0]/LA.norm(ENU),ENU[1]/LA.norm(ENU));
    if (Azimuth < 0).any():
      Azimuth = Azimuth + 2 * np.pi;
    E=np.degrees(Elevation);
    A=np.degrees(Azimuth);
    return [E,A]

##import GALILEO almanach data
sat_data = read_xml('2022-05-17.xml')


##BUTE coordinates
lat =np.radians(47.5)
lon =np.radians(19)
alt =0
##Station(/reciever) ECEF coordinates
xsta = GPS2ECEF(lat,lon,alt);

xsat = []
cutoff = 10 #Elevation cutoff
nsat = 0;
orientation = [];
zenit = [];
[gps_week, sec_of_week] = ymdhms2gps(2022,5,17,19,0,0)
for i in range(0,len(sat_data)):
    a = satpos(gps_week, sec_of_week, sat_data.iloc[i]['ID'], sat_data.iloc[i]['statusE5a'], sat_data.iloc[i]['ecc'], sat_data.iloc[i]['ax'], sat_data.iloc[i]['omega0'],\
                sat_data.iloc[i]['w'], sat_data.iloc[i]['m0'], sat_data.iloc[i]['t0a'], sat_data.iloc[i]['deltai'], sat_data.iloc[i]['omegadot'], sat_data.iloc[i]['wna'])
    xsat.append(a)
    AE = Calc_Azimuth_Elevation(xsta,xsat[i])
    zenit = (np.pi/2)-AE[1]
    if AE[0]>cutoff:
        #print('{0:.0f}'.format(sat_data.iloc[i]['ID']),AE)
        nsat +=1;
        ##Plot
        ax = plt.subplot(1, 1, 1, projection='polar')
        ax.grid(True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_yticks(range(0, 90, 10))                   # Define the yticks
        yLabel = ['90', '', '', '60', '', '', '30', '', '']
        ax.set_yticklabels(yLabel)
        ax.plot(np.radians(AE[1]),np.degrees((np.pi/2)-np.radians(AE[0])), 'bo')
        ax.text(np.radians(AE[1]),np.degrees((np.pi/2)-np.radians(AE[0])),'E' + '{0:.0f}'.format(sat_data.iloc[i]['ID']),
                horizontalalignment='center',
                verticalalignment='bottom')
plt.show()
        
print('Number of visible satellites at the mentioned moment:', nsat)







