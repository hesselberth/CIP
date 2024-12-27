#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:38:37 2024

@author: Marcel Hesselberth
"""

from constants import AS2RAD
import numpy as np
from dt import UTCTime, UT1Time
from cip import *

defaultprint = np.get_printoptions()

print()
utc = UTCTime(2006, 1, 15, 21, 24, 37.5)  # Wallace & Capitaine 2006
tt = utc.tt()
tt_mjd = tt.mjd()
print(f"Running test for transformation at {utc}.")
print(F"tt_mjd = {tt_mjd}")
ref = 53750.892855138888889
mjd_err = abs(tt_mjd - ref)
print(f"tt_mjd ERR = {mjd_err:.1e} day\n")
assert(mjd_err < 1e-10)

tjc = utc.tt().tjc()
print(f"tjc in Julian centuries = {tjc}")
ref = +0.06040774415164651
tjc_err = abs(tjc - ref)
print(f"tjc ERR: {tjc_err:.1e} Jcy\n")
assert(tjc_err < 1e-10)

ut1 = utc.ut1()
ut1mutc = ut1 - utc
print(f"ut1-utc = {ut1mutc}")
ref = .3341
ut1_err = abs(ut1mutc - ref)
print(f"ut1-utc ERR: {ut1_err:.1e} s (reference has 4 decimals)\n")
assert(ut1_err < .0001)

# continue with reference ut1mutc
ut1 = UT1Time(utc + ref) 
ut1_mjd = ut1.mjd()
print(f"ut1_mjd = {ut1_mjd}")
ref = 53750.892104561342593
ut1_mjd_err = abs(ut1_mjd - ref)
print(f"ut1_mjd ERR: {ut1_mjd_err} day\n")
assert(ut1_mjd_err < 1e-10)

X, Y, s = XYs(tjc)
Z = np.sqrt(1 - X*X - Y*Y)
print(f"X = {X/AS2RAD}")
ref = +120.635997299064
X_err = abs(X/AS2RAD - ref)
print(f"X ERR: {X_err:.1e} arcsec\n")
assert(X_err < 1e-7)

print(f"Y = {X/AS2RAD}")
ref = +8.567258740044 
Y_err = abs(Y/AS2RAD - ref)
print(f"Y ERR: {Y_err:.1e} arcsec\n")
assert(Y_err < 1e-7)

print(f"Z= {Z}")
ref = +0.999999828106893 
Z_err = abs(Z - ref)
print(f"Z ERR: {Z_err:.1e} (-)\n")
assert(Z_err < 1e-12)

print(f"s= {s/AS2RAD}")
ref = -0.002571986
s_err = abs(s/AS2RAD - ref)
print(f"s ERR: {s_err:.1e} arcsec\n")
assert(s_err < 1e-7)

print("Mcio")
mcio = Mcio(tjc)
print(mcio)
ref = [[+0.99999982896948063, +0.00000000032319161, -0.00058485982037403],
       [-0.00000002461548575, +0.99999999913741183, -0.00004153523474454],
       [+0.00058485981985612, +0.00004153524203735, +0.99999982810689262]]
print("ERR Mcio")
err = mcio - ref
np.set_printoptions(precision = 1)
print(err, '\n')
np.set_printoptions(precision = 20)
assert(np.max(np.abs(err)) < 3e-12)

era = ERA(ut1.jd2k())
print(f"ERA = {era/DEG2RAD} degrees")
ref = 76.265431053522
s_err = abs(era/DEG2RAD - ref)
print(f"ERA ERR: {s_err:.1e} degree\n")
assert(s_err < 1 / 3600e6)

print("R")
r = R(ut1.jd2k(), tjc)
print(r)
ref = [[+0.23742421473053985, +0.97140604802742432, -0.00017920749958268],
       [-0.97140588849284706, +0.23742427873021974, +0.00055827489403210],
       [+0.00058485981985612, +0.00004153524203735, +0.99999982810689262]]
print("ERR R")
err = r - ref
np.set_printoptions(precision = 1)
print(r - ref, '\n')
np.set_printoptions(precision = 20)
assert(np.max(np.abs(err)) < 3e-12)

gamma, phi  = PFW(tjc)
print(f"gamma = {gamma/AS2RAD} arcsec")
ref = +0.586558662
gamma_err = abs(gamma/AS2RAD - ref)
print(f"gamma ERR: {gamma_err:.1e} arcsec\n")
assert(gamma_err < 1e-7)

print(f"phi = {phi/AS2RAD} arcsec")
ref = +84378.585257806
phi_err = abs(phi/AS2RAD - ref)
print(f"phi ERR: {phi_err:.1e} arcsec\n")
assert(phi_err < 1e-7)

print("Mclass")
mclass, eo = Mclass_EO(tjc)
print(mclass)
ref = [[+0.99999892304984912, -0.00134606988972260, -0.00058480338056834],
       [+0.00134604536839225, +0.99999909318492665, -0.00004232245992880],
       [+0.00058485981924879, +0.00004153524246778, +0.99999982810689296]]
print("ERR Mclass")
err = mclass - ref
np.set_printoptions(precision = 1)
print(mclass - ref, '\n')
np.set_printoptions(precision = 20)
assert(np.max(np.abs(err)) < 3e-12)

print("Mclass(Mcio)")
mclass = Mclass(mcio, tjc)
print(mclass)
ref = [[+0.99999892304984912, -0.00134606988972260, -0.00058480338056834],
       [+0.00134604536839225, +0.99999909318492665, -0.00004232245992880],
       [+0.00058485981924879, +0.00004153524246778, +0.99999982810689296]]
print("ERR Mclass")
err = mclass - ref
np.set_printoptions(precision = 1)
print(mclass - ref, '\n')
np.set_printoptions(precision = 20)
assert(np.max(np.abs(err)) < 3e-12)

print(f"EO = {eo/AS2RAD} arcsec")
ref = -277.646995746
eo_err = abs(eo/AS2RAD - ref)
print(f"EO ERR: {eo_err:.1e} arcsec\n")
assert(eo_err < 1e-7)

gst = GST(ut1.jd2k(), eo) / DEG2RAD
gst /= 15
print(gst)
hh, mm = divmod(gst, 1)
mm, ss = divmod(mm*60, 1)
ss*=60
hh = int(hh)
mm = int(mm)
print(f"GST = {hh}:{mm}:{ss}")
hh_ref, mm_ref, ss_ref = 5,  5, 22.213252562
print(f"GST ERR: {hh-hh_ref}:{mm-mm_ref}:{ss-ss_ref}\n")
assert(hh == hh_ref and mm == mm_ref and (ss - ss_ref) < 1e-6)
