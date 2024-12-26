#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:28:19 2023

@author: hessel
"""

import numpy as np
from constants import AS2RAD, UAS2RAD, PI, PI2, DEG2RAD
from rot3d import *
import data2 as data

_X = data.X()
_Y = data.Y()
_S = data.s()
_X_poly_5 = data.X_poly_5
_Y_poly_5 = data.Y_poly_5
_S_poly_5 = data.s_poly_5
_powers_5 = data.powers_5
_prec_poly_5 = data.prec_poly_5


def XYS06(tjc):
    T = np.power(tjc, _powers_5)
    x = (np.dot(T, _X_poly_5) + _X(tjc)) * UAS2RAD
    y = (np.dot(T, _Y_poly_5) + _Y(tjc)) * UAS2RAD
    spxy2 = (np.dot(T, _S_poly_5) + _S(tjc)) * UAS2RAD
    return np.array([x, y, spxy2 - (x*y/2)])


def Mcio(tjc):
    M = np.empty((3,3))  # result

    X, Y, s = XYS06(tjc)
    Z = np.sqrt(1 - X*X - Y*Y)
    a = 1 / (1+Z)
    ss, cs = np.sin(s), np.cos(s)

    M[0][0] = cs + a * X * ( Y * ss - X * cs)
    M[0][1] = -ss + a * Y * (Y * ss - X * cs)
    M[0][2] = -(X * cs - Y * ss)
    M[1][0] = ss - a * X * (Y * cs + X * ss)
    M[1][1] = cs - a * Y * (Y * cs + X * ss)
    M[1][2] = -(Y * cs + X * ss)
    M[2][0] = X
    M[2][1] = Y
    M[2][2] = Z
    return M


def ERA(UT1_2000):  # UT1 is Julian UT1 date since JD2000
    T_u = UT1_2000
    f = UT1_2000 % 1
    turns = (f + 0.7790572732640 + 0.00273781191135448 * T_u) % 1
    return PI2 * turns


def R(ut1, tjc):
    return R3(ERA(ut1)) @ Mcio(tjc)


# for the classical equinox based matrix Mclass
def PFW06(tjc):  # xi, eps, gamma, phi, psi
    return np.dot(np.power(tjc, _powers_5), _prec_poly_5) * AS2RAD


# classical equinox based NPB matrix
def Mclass(tjc):
    xi, eps, gamma, phi, psi = PFW06(tjc)
    k = np.array([np.sin(phi)*np.sin(gamma), \
                  -np.sin(phi)*np.cos(gamma), \
                  np.cos(phi)])
    X, Y, s = XYS06(tjc)
    Z = np.sqrt(1 - X*X - Y*Y)
    n = np.array([X, Y, Z])
    nxk = np.cross(n, k)
    nxnxk = np.cross(n, nxk)
    nxk = nxk / np.linalg.norm(nxk)
    nxnxk = nxnxk / np.linalg.norm(nxnxk)
    Mclass = np.array([nxk, nxnxk, n], dtype=np.double)
    return Mclass
    
    
def ee06a(tjc):
    dpsi = delta_psi(tjc)
    eps_A = np.dot(np.power(tjc, powers_5), prec_poly_5[:,1])
    return dpsi*np.cos(eps_A * AS2RAD)


def eo06a(tjc):
    eo = -ee06a(tjc) * 1e-6
    T = np.power(tjc, powers_5)
    eo -= np.dot(T, GST_poly_5)
    eo -= GSTnp(tjc) * 1e-6
    return eo


def EE06a(tjc):
    dpsi = delta_psi(f, tjc)
    print(np.power(tjc, powers_5))
    eps_A = np.dot(np.power(tjc, powers_5), prec_poly_5[:,1])
    return dpsi*np.cos(eps_A * AS2RAD)


def EO06a(tjc):
    eo = -ee06a(tjc) * 1e-6
    T = np.power(tjc, powers_5)
    eo -= np.dot(T, GST_poly_5)
    eo -= GSTnp(tjc) * 1e-6
    return eo


def GST(UT1, tjc):
        return ERA(UT1) - eo06a(tjc)


def eo06(tjc):
    X, Y, s = xys06(tjc) * AS2RAD
    Z = np.sqrt(1 - X*X - Y*Y)
    a = 1 / (1 + Z)    
    xi, eps, gamma, phi, psi = prec06(tjc)
    dpsi, deps = nut06a(tjc) * 1e-6
    psi += dpsi
    eps += deps
    spsi, cpsi = np.sin(psi*AS2RAD), np.cos(psi*AS2RAD)
    sgam, cgam = np.sin(gamma*AS2RAD), np.cos(gamma*AS2RAD)
    sphi, cphi = np.sin(phi*AS2RAD), np.cos(phi*AS2RAD)
    seps, ceps = np.sin(eps*AS2RAD), np.cos(eps*AS2RAD)
    A = np.empty(3, dtype=np.double)
    y = np.empty(3, dtype=np.double)
    S = np.empty(3, dtype=np.double)
    A[0] = cpsi*cgam + spsi*cphi*sgam
    A[1] = cpsi*sgam - spsi*cphi*cgam
    A[2] = -spsi*sphi
    y[0] = ceps*spsi*cgam - (ceps*cpsi*cphi + seps*sphi) * sgam
    y[1] = ceps*spsi*sgam + (ceps*cpsi*cphi + seps*sphi) * cgam
    y[2] = ceps*cpsi*sphi - seps*cphi
    S[0] = 1 - X*X*a
    S[1] = -X*Y*a
    S[2] = -X
    return (s - np.arctan2(np.dot(y, S), np.dot(A, S)))/AS2RAD
    



if __name__ == "__main__":
    from dt import UTCTime, UT1Time
    t = UTCTime(2006, 1, 15, 21, 24, 37.5)
    print("tt", t.tt().jd() - 2400000.5)
    tjc = t.tdb().tjc()
    X, Y, s = XYS06(tjc)
    Z = np.sqrt(1 - X*X - Y*Y)
    X /= AS2RAD
    ref = +120.635997299064
    print("X", X)
    print("ERR X", X - ref)
    Y /= AS2RAD
    ref = +8.567258740044 
    print("Y", Y)
    print("ERR Y", Y - ref)
    ref = +0.999999828106893 
    print("Z", Z)
    print("ERR Z", Z - ref)
    s /= AS2RAD
    ref = -0.002571986
    print("s", s)
    print("ERR s", s - ref)
    print()
    print("Mcio")
    print(Mcio(tjc))
    ref = [[+0.99999982896948063, +0.00000000032319161, -0.00058485982037403],
           [-0.00000002461548575, +0.99999999913741183, -0.00004153523474454],
           [+0.00058485981985612, +0.00004153524203735, +0.99999982810689262]]
    print("ERR Mcio")
    print(Mcio(tjc) - ref)
    print()
    ut1 = UT1Time(t+.3341).jd2k()
    #ut1 = t.ut1().jd2k()
    era = ERA(ut1) / DEG2RAD
    print("ERA", era)
    ref = 76.265431053522
    print("ERR ERA", era - ref)
    print()
    print("R")
    print(R(ut1, tjc))
    ref = [[+0.23742421473053985, +0.97140604802742432, -0.00017920749958268],
           [-0.97140588849284706, +0.23742427873021974, +0.00055827489403210],
           [+0.00058485981985612, +0.00004153524203735, +0.99999982810689262]]
    print("ERR R")
    print(R(ut1, tjc) - ref)
    print()