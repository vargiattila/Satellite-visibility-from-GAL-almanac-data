import math
import re
import matplotlib.pyplot as plt
import pandas as pd
import sys
from itertools import takewhile, islice, dropwhile

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
            ID.append(str(result.group(1)))
        elif re.match('<aSqRoot>', line):
            result = re.search('<aSqRoot>(.*)</aSqRoot>', line)
            aSqRoot.append(str(result.group(1)))
        elif re.match('<ecc>', line):
            result = re.search('<ecc>(.*)</ecc>', line)
            ecc.append(str(result.group(1)))
        elif re.match('<deltai>', line):
            result = re.search('<deltai>(.*)</deltai>', line)
            deltai.append(str(result.group(1)))
        elif re.match('<omega0>', line):
            result = re.search('<omega0>(.*)</omega0>', line)
            omega0.append(str(result.group(1)))
        elif re.match('<omegaDot>', line):
            result = re.search('<omegaDot>(.*)</omegaDot>', line)
            omegaDot.append(str(result.group(1)))
        elif re.match('<w>', line):
            result = re.search('<w>(.*)</w>', line)
            w.append(str(result.group(1)))
        elif re.match('<m0>', line):
            result = re.search('<m0>(.*)</m0>', line)
            m0.append(str(result.group(1)))
        elif re.match('<af0>', line):
            result = re.search('<af0>(.*)</af0>', line)
            af0.append(str(result.group(1)))
        elif re.match('<af1>', line):
            result = re.search('<af1>(.*)</af1>', line)
            af1.append(str(result.group(1)))
        elif re.match('<iod>', line):
            result = re.search('<iod>(.*)</iod>', line)
            iod.append(str(result.group(1)))
        elif re.match('<t0a>', line):
            result = re.search('<t0a>(.*)</t0a>', line)
            t0a.append(str(result.group(1)))
        elif re.match('<wna>', line):
            result = re.search('<wna>(.*)</wna>', line)
            wna.append(str(result.group(1)))
        elif re.match('<statusE5a>', line):
            result = re.search('<statusE5a>(.*)</statusE5a>', line)
            statusE5a.append(str(result.group(1)))
        elif re.match('<statusE5b>', line):
            result = re.search('<statusE5b>(.*)</statusE5b>', line)
            statusE5b.append(str(result.group(1)))
        elif re.match('<statusE1B>', line):
            result = re.search('<statusE1B>(.*)</statusE1B>', line)
            statusE1B.append(str(result.group(1)))
            
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
    return ID,aSqRoot,ecc,deltai,omega0,omegaDot,w,m0,af0,af1,iod,t0a,wna,statusE5a 

sat_data = read_xml('2022-05-06.xml')

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
    sec_of_week = ((mjd - 44244) - gps_week * 7 + fmjd) * 86400

    return gps_week,sec_of_week

[gps_week, sec_of_week] = ymdhms2gps(2022, 5, 6, 8, 34, 0)
print(sec_of_week)
print(sat_data[11][1])

def satpos(gps_week, sec_of_week, prn, health, ecc, ax, raw, aop, man, toa, inc, rra, week):
    xsat = []
    Wedot = 7.2921151467e-5;	#WGS 84 value of earth's rotation rate
    mu =  3.986005e+14;		#WGS 84 value of earth's univ. grav. par.
    #mean motion
    n = math.sqrt(mu / float(ax)**6);
    T = 2.0 * math.pi / n;
    dt = sec_of_week - float(toa)*604800;
    if abs(dt) > 604800:
        print('*** to much time difference %f\n', dt);
    else:
        M = float(man) + n * dt;
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
    #position in orbit plane
        u = nu+float(aop)
        r = float(ax)*(1.0-float(ecc)*math.cos(E))
        wc = float(raw)+(float(rra)-Wedot)*dt-float(toa)*Wedot
        xdash = r*math.cos(u)
        ydash = r*math.sin(u)
    #position in ECEF system
        xsat.append(float(xdash)*math.cos(float(wc)) - float(ydash)*math.cos(float(inc))*math.sin(float(wc)))
        xsat.append(xdash*math.sin(float(wc)) + ydash*math.cos(float(inc))*math.cos(float(wc)))
        xsat.append(ydash*math.sin(float(inc)))
    return xsat

print(satpos(gps_week, sec_of_week,sat_data[0][1],sat_data[13][1],sat_data[2][1],sat_data[1][1],sat_data[4][1],\
             sat_data[6][1],sat_data[7][1],sat_data[11][1],sat_data[3][1],sat_data[5][1],sat_data[12][1]))

