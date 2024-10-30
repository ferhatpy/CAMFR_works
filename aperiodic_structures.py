#!/usr/bin/env python
# -*- coding: iso-8859-9 -*-

####################################################################
#
# Propagation of light in aperiodic structures.
# 
# References:
#
#
#
####################################################################
import numpy as np
from camfr import *
#from sympy import *

"""
# structure
capless -> stack of layers (planar or slab) without air and substrate cap layers.
capped  -> stack of layers (planar or slab) with air and substrate cap layers.

# materials
2_layers -> is developed for testing Brewster's angle.

# calculation
In R_vs_l_vs_theta calculation, at each iteration of lambda,
set_lambda(l)->set_material() routine is executed.
This changes current lambda and quarter-wave optical thicknesses.


# Some useful codes of CAMFR.
plot_n(s,arange(0,5,1),arange(0,100,0.5))
r_x = arange(0,10,1)
r_z = arange(0,100,1) 
s.plot_field(s, lambda f : f.E1(), vx, vz)

# Some useful codes of this code.
Fibonacci(n=2, conj=False, text=True) => 

# References: 
- CsBr_Te_Ge, Del Barco, O. et al. (2017) Omnidirectional high-reflectivity mirror in the 4-20 um spectral range, Journal of Optics (United Kingdom). IOP Publishing, 19(6). doi: 10.1088/2040-8986/aa6c76.
"""
### Settings
"""
Omnidirectional Mirror
sol:1 (planar), structure:32 (capped_RS_cRS), materials:40 (TiO2_SiO2),
simulation:5 (R_vs_l_vs_theta)

Fiber Bragg Gratings (FBGs)
sol:4 (circ), structure:2 (capless_Fib), materials:1 (GaAs_AlAs), 
simulation:3 (R_vs_l)
"""

lang = {1:"tr", 2:"en"}[2]
sol  = {1:"planar",     2:"slab",
        3:"slabfiber",  4:"circ"}[4]
structure = {2:"capless_Fib", 4:"capped_Fib",
             6:"capless_TM",  8:"capped_TM",
            10:"capless_DP",  12:"capped_DP",
            14:"capless_RS",  16:"capped_RS",
            18:"capless_PR",  19:"capless_PR_cPR",
            20:"capped_PR",   21:"capped_PR_cPR",
            22:"capless_Fib_cFib", 24:"capped_Fib_cFib",
            25:"",                 26:"capped_TM_cTM",
            27:"",                 28:"capped_DP_cDP",
            30:"capless_RS_cRS",   32:"capped_RS_cRS",
            40:"sarcan_fig_2_16",  42:"2_layers",
            44:"ghatak",
            50:"Barco2017_Fig2b", 52:"Barco2017_Fig2d"}[18]
materials = {1:"GaAs_AlAs", 20:"2_layers", 21:"ghatak", 
            30:"Barco2017_Fig2b", 31:"Barco2017_Fig2d",
            40:"TiO2_SiO2"}[1]
simulation = {1:"R_vs_l_const_n", 2:"R_vs_theta_const_n",
              3:"R_vs_l", 4:"",
              5:"R_vs_l_vs_theta", 6:"n_vs_l"}
#simulation = simulation.values() # Execute all simulations
simulation = simulation[3]

if materials in ["TiO2_SiO2"]:
    (target_l, min_l, max_l, dl) = (0.6, 0.43, 1.535, 0.002) # 0.002 Wavelength min_lambda, max_lambda, delta_lambda in (um).
else:
    (target_l, min_l, max_l, dl) = (1.3, 1.1, 1.5, 0.001) # SakineThesis, Wavelength min_lambda, max_lambda, delta_lambda in (um).

if structure in ["Barco2017_Fig2b", "Barco2017_Fig2d"]:
    (target_l, min_l, max_l, dl) = (4, 4, 20, 0.01) # 0.01 min_lambda, max_lambda, delta_lambda in (um).

set_lambda(target_l)  # Set design wavelength of the DBR structure.
eps = 1e-5            # Precision for calculating overlap integrals. (1e-5)
T = 4                 # 5 Period of the Fibonacci cluster. Fibonacci=20
order = 2             # Order of the Fibonacci replacement sequence. Fibonacci=5

# Lists of R_l, R_l_theta calculations.
[ls, ns, n_vs_l, RsTE, RsTM, R_vs_l, R_vs_l_vs_theta] = [[],[],[],[],[],[],[]]
# Lists of R_theta calculation.
[thetas, RsTE_t, RsTM_t, R_vs_theta] = [[],[],[],[]]
filename = "fib_1D_sol_{0}_mat_{1}_ord{2}_T{3}_str_{4}"\
.format(sol, materials, order, T, structure)
print(filename)

"""
# Optical constants of GaAs (Gallium arsenide)
# todo add from reflectance.info write in references
"""
class CsBr_Factory:
  def __call__(self):
    # Barco2017, Sellmeier-type dispersion equation.
    [A1, B1, C1, B2, C2, B3, C3] = [1, 0.953, 0.008, 0.830, 0.027, 2.847, 14164.689]
    l = get_lambda().real
    n = sqrt(A1 + B1*l**2/(l**2 - C1) + B2*l**2/(l**2 - C2) + B3*l**2/(l**2 - C3)) 
    return Material(n)

class Ge_Factory:
  def __call__(self):
    # Barco2017, Sellmeier-type dispersion equation.
    [A1, B1, C1, B2, C2, B3, C3] = [9.282, 6.729, 0.441, 0.213, 3870.100, 0.0, 0.0]
    l = get_lambda().real
    n = sqrt(A1 + B1*l**2/(l**2 - C1) + B2*l**2/(l**2 - C2) + B3*l**2/(l**2 - C3)) 
    return Material(n)

class GlassBK7_Factory:
  def __call__(self):
    # SCHOTT Zemax catalog 2017-01-20b (obtained from http://www.schott.com).
    [A1, B1, C1, B2, C2, B3, C3] = [1, 1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653]
    l = get_lambda().real
    n = sqrt(A1 + B1*l**2/(l**2 - C1) + B2*l**2/(l**2 - C2) + B3*l**2/(l**2 - C3)) 
    return Material(n)

class SiO2_Factory:
  def __call__(self):
    """
    Crystal n_ordinary, l=[0.198, 2.0531]
    G. Ghosh. Dispersion-equation coefficients for the refractive index and birefringence of calcite and quartz crystals, Opt. Commun. 163, 95-102 (1999), Sellmeier-type dispersion equation.
    """
    [A1, B1, C1, B2, C2, B3, C3] = [1+0.28604141, 1.07044083, 1.00585997E-2, 1.10202242, 100, 0.0, 0.0]
    l = get_lambda().real
    n = sqrt(A1 + B1*l**2/(l**2 - C1) + B2*l**2/(l**2 - C2) + B3*l**2/(l**2 - C3)) 
    return Material(n)
    
class TiO2_Factory:
  def __call__(self):
    """
    The refractive indices used are nA=1.46 and nB=2.42,
    which represent the materials SiO2 and TiO2, 
    at the wavelength of 500 nm, respectively.
    
    Crystal n_ordinary, l=[0.43, 1.53]
    J. R. Devore. Refractive indices of rutile and sphalerite, J. Opt. Soc. Am. 41, 416-419 (1951), Sellmeier-type dispersion equation.
    """
    [A1, B1, C1, B2, C2, B3, C3] = [5.913, 0.2441, 0.0803, 0.0, 0.0, 0.0, 0.0]
    l = get_lambda().real
    n = sqrt(A1 + B1/(l**2 - C1) + B2*l**2/(l**2 - C2) + B3*l**2/(l**2 - C3)) 
    return Material(n)

class ZnS_Factory:
  def __call__(self):
    # Barco2017
    eps = 5.164 + 0.1208 / (get_lambda().real **2 - 0.27055**2)
    return Material(sqrt(eps))


"""
Functions Section
"""
def DoublePeriod(n, conj=False, text=False):
    """
    Generates a double-period replacement sequence.
    A -> AB, B -> AA.
    """
    if text == False:
        if conj==False:
            # Double-period sequence.
            a, b = A(d_A), B(d_B)
        else:
            # Conjugate double-period sequence.
            a, b = B(d_B), A(d_A)
        for i in range(0, n):
            a, b = a+b, a+a
        return a
    
    elif text == True:
        if conj==False:
            # Double-period sequence.
            a, b = "A","B"
        else:
            # Conjugate double-period sequence.
            a, b = "B","A"
        for i in range(0, n):
            print("DP({0}) = {1}".format(i,a))
            a, b = a+b, a+a
        print("DP({0}) = {1}".format(n,a))

def Fibonacci(n, conj=False, text=False):
    """
    Generates a Fibonacci replacement sequence.
    A -> AB, B -> A.
    """
    if text == False:
        if conj==False:
            # Fibonacci sequence
            a, b = B(d_B), A(d_A)
        else:
            # Conjugated Fibonacci sequence
            a, b = A(d_A), B(d_B)
        for i in range(0, n):
            a, b = b, b+a
        return a
    
    elif text == True:
        if conj==False:
            # Fibonacci sequence
            a, b = "B","A"
        else:
            # Conjugated Fibonacci sequence
            a, b = "A","B"
        for i in range(0, n):
            print("F({0}) = {1}".format(i,a))
            a, b = b, b+a
        print("F({0}) = {1}".format(n,a))
        
def Periodic(n, conj=False, text=False):
    """
    Generates a Periodic sequence.
    A -> AB, B -> AB
    """
    if text == False:
        if conj==False:
            # Periodic sequence
            a, b = A(d_A), B(d_B)
        else:
            # Conjugated Periodic sequence
            a, b = B(d_B), A(d_A)
        for i in range(0, n):
            a, b = a+b, a+b
        return a
    
    elif text == True:
        if conj==False:
            # Periodic sequence
            a, b = "A","B"
        else:
            # Conjugated Periodic sequence
            a, b = "B","A"
        for i in range(0, n):
            print("PR({0}) = {1}".format(i,a))
            a, b = a+b, a+b
        print("PR({0}) = {1}".format(n,a))        
        
def RudinShapiro(n, conj=False, text=False):
    """    
    Generates a Rudin-Shapiro replacement sequence.
    AA -> AAAB, AB -> AABA
    BA -> BBAB, BB -> BBBA
    """
    if text == False:
        if conj==False:
            # RudinShapiro sequence
            aa, ab = A(d_A)+A(d_A), A(d_A)+B(d_B)
            ba, bb = B(d_B)+A(d_A), B(d_B)+B(d_B)
        else:
            # Conjugated RudinShapiro sequence
            aa, ab = B(d_B)+B(d_B), B(d_B)+A(d_A)
            ba, bb = A(d_A)+B(d_B), A(d_A)+A(d_A)
        for i in range(1, n):
            aa, ab, ba, bb = aa+ab, aa+ba, bb+ab, bb+ba
        return aa
    
    elif text == True:
        if conj==False:
            # RudinShapiro sequence
            aa, ab = "AA", "AB"
            ba, bb = "BA", "BB"
        else:
            # Conjugated RudinShapiro sequence
            aa, ab = "BB", "BA"
            ba, bb = "AB", "AA"
        for i in range(1, n):
            print("RS({0}) = {1}".format(i, aa))
            aa, ab, ba, bb = aa+ab, aa+ba, bb+ab, bb+ba
        print("RS({0}) = {1}".format(n, aa))
        
def ThueMorse(n, conj=False, text=False):
    """    
    Generates a Thue-Morse replacement sequence.
    A -> AB, B -> BA.
    """
    """
    # length of each unit
    nh = A_m.n()
    nl = B_m.n()
    n0 = nh-(nh-nl)/2.0       # no=(nh+nl)/2, Deltan=nh-n0=n0-nl
    d_A = (get_lambda()/4./n0).real
    d_B = (get_lambda()/4./n0).real
    """
    if text == False:
        if conj==False:
            # ThueMorse sequence.
            a, b = A(d_A), B(d_B)
        else:
            # Conjugated ThueMorse sequence.
            a, b = B(d_B), A(d_A)
        for i in range(0, n):
            a, b = a+b, b+a            
        return a
    
    elif text == True:
        if conj==False:
            # ThueMorse sequence.
            a, b = "A","B"
        else:
            # Conjugated ThueMorse sequence.
            a, b = "B","A"
        for i in range(0, n):
            print("TM({0}) = {1}".format(i,a))
            a, b = a+b, b+a
        print("TM({0}) = {1}".format(n,a))

def set_material():
    global A_m, B_m, C_m, air_m
    global clad_m
    global d_A, d_B, d_C
    global target_l
    
    if materials == "GaAs_AlAs":
        # A = GaAs ve B = AlAs
        A_m = Material(3.4059)    # GaAs
        B_m = Material(2.9086)    # AlAs
        C_m = Material(3.4059)    # GaAs
        clad_m = Material(2.90)   # Cladding material.
        air_m  = Material(1.0)    # Air
    
        # length of each unit
        """
        d_A = (get_lambda()/4./A_m.n().real).real
        d_B = (get_lambda()/4./B_m.n().real).real
        will produce a flat R_lambda because n also changes with lambda.
        """
        # Quarter-wavelength condition.
        d_A = (target_l/4./A_m.n().real).real
        d_B = (target_l/4./B_m.n().real).real
        d_C = 0
        
    elif materials == "TiO2_SiO2":
        # TiO2_SiO2, n(TiO2) > n(SiO2)
        A_m = TiO2_Factory()()  # TiO2
        B_m = SiO2_Factory()()  # SiO2
        C_m = GlassBK7_Factory()() # BK7 Glass
        air_m  = Material(1.0)  # Air
        
        # length of each unit
        """
        d_A = (get_lambda()/4./A_m.n().real).real
        d_B = (get_lambda()/4./B_m.n().real).real
        will produce a flat R_lambda because n also changes with lambda.
        """
        # Quarter-wavelength condition.
        d_A = (target_l/4./A_m.n().real).real
        d_B = (target_l/4./B_m.n().real).real
        d_C = 0
        
        """
        # Half-wavelength condition.
        d_A = (target_l/2./A_m.n().real).real # d_A < d_B
        d_B = (target_l/B_m.n().real).real
        d_C = 0
        """
        
        """
        n_eff = 2.0
        d_A = (target_l/4./n_eff).real
        d_B = (target_l/4./n_eff).real
        d_C = 0
        """

    elif materials == "2_layers":
        # 2_layers is developed for testing Brewster's angle.
        target_l=0.5876
        A_m = Material(3.9476)
        B_m = Material(3.9476)
        C_m = Material(3.9476)
        air_m  = Material(1.0)

    elif materials == "ghatak":
        # Ghatak p203, ch15.6
        A_m = Material(3.4059)    # GaAs
        B_m = Material(2.9086)    # AlAs
        C_m = Material(3.4059)    # GaAs
        clad_m = Material(2.90)   # Cladding material.
        air_m  = Material(1.0)    # Air
        nh = A_m.n()
        nl = B_m.n()
        n0 = nh-(nh-nl)/2.0       # no=(nh+nl)/2, Deltan=nh-n0=n0-nl
        d_A = get_lambda()/4./n0
        d_B = d_A

    elif materials == "Barco2017_Fig2b":
        # CsBr_Te_Ge, n(Te) > n(CsBr)
        A_m = CsBr_Factory()()  # CsBr
        B_m = Material(5.3)     # Te
        C_m = Ge_Factory()()    # Ge
        air_m  = Material(1.0)  # Air
        [d_A, d_B, d_C] = [2.5301, 0.7925, 20]

    elif materials == "Barco2017_Fig2d":
        # CsBr_Te_Ge, n(Te) > n(CsBr)
        A_m = CsBr_Factory()()  # CsBr
        B_m = Material(5.3)     # Te
        C_m = Ge_Factory()()    # Ge
        air_m  = Material(1.0)  # Air
        [d_A, d_B, d_C] = [3.2229, 1.0094, 20]
        

def set_structure():
    global T, order

    if structure == "capless_PR":
        # res = Stack(T*(A(d_A)+B(d_B)))
        res = Stack(T*Periodic(order))
    
    elif structure == "capped_PR":
        res = Stack(air(0) + T*Periodic(order) + C(d_C))
        
    elif structure == "capped_PR_cPR":
        res = Stack(air(0) + T*(Periodic(order) + Periodic(order, conj=True)) + C(d_C))    
    
    elif structure == "capless_Fib":
        res = Stack(T*Fibonacci(order))
        
    elif structure == "capped_Fib":
        res = Stack(air(0) + T*Fibonacci(order) + C(d_C))

    elif structure == "capless_Fib_cFib":
        res = Stack(T*(Fibonacci(order) + Fibonacci(order, conj=True)))

    elif structure == "capped_Fib_cFib":
        res = Stack(air(0) + T*(Fibonacci(order) + Fibonacci(order, conj=True)) + C(d_C))    
    
    elif structure == "capless_TM":
        res = Stack(T*ThueMorse(order))
        
    elif structure == "capped_TM":
        res = Stack(air(0) + T*ThueMorse(order) + C(d_C))

    elif structure == "capped_TM_cTM":
        res = Stack(air(0) + T*(ThueMorse(order) + ThueMorse(order, conj=True)) + C(d_C))    
    
    elif structure == "capless_DP":
        res = Stack(T*DoublePeriod(order))
        
    elif structure == "capped_DP":
        res = Stack(air(0) + T*DoublePeriod(order) + C(d_C))
        
    elif structure == "capped_DP_cDP":
        res = Stack(air(0) + T*(DoublePeriod(order) + DoublePeriod(order, conj=True)) + C(d_C))
        
    elif structure == "capless_RS":
        res = Stack(T*RudinShapiro(order))
        
    elif structure == "capped_RS":
        res = Stack(air(0) + T*RudinShapiro(order) + C(d_C))

    elif structure == "capped_RS_cRS":
        res = Stack(air(0) + T*(RudinShapiro(order) + RudinShapiro(order, conj=True)) + C(d_C))

    elif structure == "sarcan_fig_2_16":
        (GaAs, AlAs)=(A, B)
        (d_GaAs, d_AlAs)=(d_A, d_B)
        res = Stack(air(0) + 25*(GaAs(d_GaAs) + AlAs(d_AlAs)) + GaAs(d_C))    
    
    elif structure == "ghatak":
        res = Stack(T*(A(d_A)+B(d_B)))

    elif structure == "2_layers":
        GaAs = A
        res = Stack(air(0)+GaAs(0))

    elif structure == "Barco2017_Fig2b":
        T = 1 
        order = 9 # Total Fibonacci(9)= 55 layers. l_total = 80 um.
        res = Stack(air(0) + T*Fibonacci(order) + C(d_C))    
    
    elif structure == "Barco2017_Fig2d":
        T = 19 # Total 2x19 = 38 layers. l_total = 80 um.
        res = Stack(air(0) + T*(A(d_A)+B(d_B)) + C(d_C))
        
    return res

def calc_R_vs_l():
    # Calculates reflectance and transmittance.
    ls.append(l)
    
    set_polarisation(TE)
    s.calc()
    refTE = abs(s.R12(0,0))**2 # R12(i,j) this is the reflection from mode j to mode i
    RsTE.append(refTE)

    set_polarisation(TM)
    s.calc()
    refTM = abs(s.R12(0,0))**2
    RsTM.append(refTM)

    R_vs_l.append([l, refTE, refTM])
    print(l, refTE, refTM)
    
def calc_R_vs_theta():
    # Calculates reflectance versus incident angle.
    # Loop over incidence angle at a fixed wavelength lambda.
    set_lambda(target_l)
    for theta in arange(0.0, 90.0, 0.01):
        thetas.append(theta)        
        air.set_theta(theta * pi / 180.)

        set_polarisation(TE)
        s.calc()
        refTE = abs(s.R12(0,0))**2
        RsTE_t.append(refTE)
    
        set_polarisation(TM)
        s.calc()
        refTM = abs(s.R12(0,0))**2
        RsTM_t.append(refTM)
    
        R_vs_theta.append([theta, refTE, refTM])    
        print(theta, refTE, refTM)

    # Find Brewster angle.
    bangle = thetas[RsTM_t.index(min(RsTM_t))]
    print "Brewster Angle=", bangle

def calc_R_vs_l_vs_theta():
    """    
    Calculates reflectance versus wavelength versus incident angle.
    Loop over incidence angle at a fixed wavelength lambda.
    For each wavelength fill a res list and return it.
    np.savetxt writes data step by step for each wavelength
    and for all swept theta angles.
    """   
    l = get_lambda().real
    for theta in arange(0, 90, 0.1):
        air.set_theta(theta * pi / 180.)

        set_polarisation(TE)
        s.calc()
        refTE = abs(s.R12(0,0))**2
    
        set_polarisation(TM)
        s.calc()
        refTM = abs(s.R12(0,0))**2
    
        R_vs_l_vs_theta.append([l, theta, refTE, refTM])
    print(l, theta, refTE, refTM)
    
def fanimate_fields():
    inc = zeros(N())
    inc[0] = 1
    s.set_inc_field(inc)
    s.calc()
    
    # Do some plotting.
    r_x = arange(-1.0, 1.0, 0.05)
    r_z = arange(-1.0, 1.0, 0.05)
    
    A.plot_n(r_x)
    A.mode(0).plot_field(lambda f : f.E2().real, r_x)
    s.plot_field(lambda f : f.E2().real, r_x, r_z)
    s.animate_field(lambda f : f.E2(), r_x, r_z)
    
#==============================================================================
#     r_x = arange(0,10,1)
#     r_z = arange(0,100,1)    
#     #animate_field(s, lambda f : f.E1(), vx, vz)
#     
#     s.plot_field(lambda f : f.E2().real, r_x, r_z)
#     s.animate_field(lambda f : f.E2(), r_x, r_z)
#==============================================================================

def plot_n_vs_l():
    # Plot refractive index versus wavelength.
    fig, ax = plt.subplots()
    ax.plot(ls, ns)
    ax.set_xlabel(txt['wavelength'])
    ax.set_ylabel(txt['n'])
    plt.title(r"n-$\lambda$")
    plt.show()
    plt.savefig("n_l_"+filename+".pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches = 0.05)

def plot_R_vs_l():
    # Plot reflectance versus wavelength.
    fig, ax = plt.subplots()
    ax.plot(ls, RsTE)
    ax.set_xlabel(txt['wavelength'])
    ax.set_ylabel(txt['reflectance'])
    plt.title("TE")
    plt.show()
    plt.savefig("R_l_"+filename+"_TE.pdf",format='pdf',dpi=300, bbox_inches='tight',pad_inches = 0.05)
    
    fig, ax = plt.subplots()
    ax.plot(ls, RsTM)
    ax.set_xlabel(txt['wavelength'])
    ax.set_ylabel(txt['reflectance'])
    plt.title("TM")
    plt.show()
    plt.savefig("R_l_"+filename+"_TM.pdf",format='pdf',dpi=300, bbox_inches='tight',pad_inches = 0.05)  
    
def plot_R_vs_theta():
    # Plot reflectance versus angle of incidence.
    fig, ax = plt.subplots()
    ax.plot(thetas, RsTE_t)
    ax.set_xlabel(txt['incidence'])
    ax.set_ylabel(txt['reflectance'])
    plt.title("TE")
    plt.show()
    plt.savefig("R_t_"+filename+"_TE.pdf",format='pdf', dpi=300, bbox_inches='tight',pad_inches = 0.05)
    
    fig, ax = plt.subplots()
    ax.plot(thetas, RsTM_t)
    ax.set_xlabel(txt['incidence'])
    ax.set_ylabel(txt['reflectance'])
    plt.title("TM")
    plt.show()
    plt.savefig("R_t_"+filename+"_TM.pdf",format='pdf', dpi=300, bbox_inches='tight',pad_inches = 0.05)    

def plot_save():
    # Plot & save results.
    if simulation in ["R_vs_l", "R_vs_l_const_n"]:
        plot_R_vs_l()
        np.savetxt("R_l_" + filename + ".txt", R_vs_l)

    elif simulation in ["R_vs_theta_const_n"]:
        plot_R_vs_theta()
        np.savetxt("R_t_" + filename + ".txt", R_vs_theta)
    
    elif simulation in ["R_vs_l_vs_theta"]:
        pass
    
    elif simulation in ["n_vs_l"]:
        plot_n_vs_l()
        np.savetxt("n_l_" + filename + ".txt", n_vs_l)
    

def set_lang(plang):
    dictionary = {
    'en': {
    'incidence'     : u"Angle of incidence $\\theta_i$",
    'reflectance'   : u"Reflectance $R=|r^2|$",
    'wavelength'    : u"Wavelength ($\mu m$)",
    'n'             : u"n"},

    'tr': {
    'incidence'     : u"Geliş açısı $\\theta_i$",
    'reflectance'   : u"Yansıtma $R=|r^2|$",
    'wavelength'    : u"Dalgaboyu ($\mu m$)",
    'n'             : u"n"},
    }
    res = dictionary[plang]
    return res    

###############################################################################
#   MAIN Simulation Section
###############################################################################

txt = set_lang(lang)

### SOLUTION: planar
if sol=="planar":
    """
    Loop over incidence wavelengths.
    Note that you should set the wavelength before setting the angle, 
    as set_theta effectively sets the transverse part of the wavevector, 
    which is dependent on the wavelength.
    """

### k_vs_w todo
    if "k_vs_w" in [simulation]:
        for l in arange(min_l, max_l, dl):
            set_lambda(l)
            set_N(1)
            set_material()
    
            A = Planar(A_m) # todo can be included in set_structure()
            B = Planar(B_m)
            C = Planar(C_m)
            air = Planar(air_m)
           
            s = set_structure()
            theta = 0 #0
            air.set_theta(theta * pi / 180.)
            #air.set_theta(theta)
            calc_R_vs_l()           

### R_vs_l_const_n
    if "R_vs_l_const_n" in [simulation]:
        set_material()
        A = Planar(A_m)
        B = Planar(B_m)
        C = Planar(C_m)
        air = Planar(air_m)
        s = set_structure()
        
        for l in arange(min_l, max_l, dl):
            set_lambda(l)
            set_N(1)    # sets the number of modes used in the series expansion.
            theta = 0   # set the incident angle. 
            air.set_theta(theta * pi / 180.)
            calc_R_vs_l()

### R_vs_l            
    if "R_vs_l" in [simulation]:
        for l in arange(min_l, max_l, dl):
            set_lambda(l)
            set_N(1)
            set_material()
    
            A = Planar(A_m) # todo can be included in set_structure()
            B = Planar(B_m)
            C = Planar(C_m)
            air = Planar(air_m)
           
            s = set_structure()
            theta = 0 #0
            air.set_theta(theta * pi / 180.)
            #air.set_theta(theta)
            calc_R_vs_l()

### R_vs_theta_const_n    
    if "R_vs_theta_const_n" in [simulation]:
        set_material()
        A = Planar(A_m)
        B = Planar(B_m)
        C = Planar(C_m)
        air = Planar(air_m)
        s = set_structure()
        calc_R_vs_theta()

### R_vs_l_vs_theta        
    if "R_vs_l_vs_theta" in [simulation]:
        # Loop over wavelengths.
        f = open("R_l_t_" + filename + ".dat", "w")
        #f = open("R_l_t_" + filename + ".npy", "w") # Opens a file for appending in binary format.
        for l in arange(min_l, max_l, dl): # Iterate over wavelengths.
            set_lambda(l)
            set_N(1)
            set_material()
    
            A = Planar(A_m) # todo can be included in set_structure()
            B = Planar(B_m)
            C = Planar(C_m)
            air = Planar(air_m)
           
            s = set_structure()
            calc_R_vs_l_vs_theta()
            free_tmps()
            #f.write("\n")
            np.savetxt(f, R_vs_l_vs_theta) # try .gz extension.
            #np.save(f, R_vs_l_vs_theta) # Save an array to a binary file in NumPy .npy format
            #np.savez(f, R_vs_l_vs_theta) # Save several arrays into an uncompressed .npz archive
            #np.savez_compressed(f, R_vs_l_vs_theta) # Save several arrays into a compressed .npz archive
            R_vs_l_vs_theta = []
        f.close()

### n_vs_l        
    if "n_vs_l" in [simulation]:
        print("Wavelenght(um) \t n")
        for l in arange(min_l, max_l, dl):
            set_lambda(l)
            #A_m = ZnS_Factory()
            #A_m = CsBr_Factory()()
            A_m = [GlassBK7_Factory()(), 
                   SiO2_Factory()(), TiO2_Factory()()][2]
            A = Planar(A_m)
            n = A_m.n().real
            ls.append(l)            
            ns.append(n)
            n_vs_l.append([l,n])
            
            print(l, n)
        n_avg = np.mean(ns)
        print("Average n=", n_avg)

### SOLUTION: slab
elif sol=="slab":
    # tutorial2.py, tutorial3.py are used.

    # Define waveguide sections.
    set_lower_PML(-0.1)
    set_upper_PML(-0.1)
    t = 100 # thickness of each slab (um). If t>>wavelength solution approaches to planar.
   
    """
    # Below code gives unnormalized reflectance values greater than 1.
    # Its geometry is a fiber.
    A = Slab(air_m(t) + A_m(t) + air_m(t))
    B = Slab(air_m(t) + B_m(t) + air_m(t))
    air = Slab(air_m(3*t))
    """    
    
    A = Slab(A_m(3*t))
    B = Slab(B_m(3*t))
    C = Slab(C_m(3*t))
    air = Slab(air_m(3*t))       
    
    s = set_structure()
     
    #s.calc()
    #s.plot()

    # Loop over incidence wavelengths.
    for l in arange(min_l, max_l, dl):
        set_lambda(l) # new
        set_N(1) # sets the number of modes used in the series expansion.
        
        # Plane wave with amplitude I and angle theta (radians).
        II = 1.0
        theta = 0.0
        s.set_inc_field_plane_wave(II, theta*pi/180, eps)
        
        # set_lambda(l) old
        calc_R_vs_l()

    #fanimate_fields()
    
### SOLUTION: slabfiber
elif sol=="slabfiber":
    # tutorial2.py, tutorial3.py are used.

    # Define waveguide sections.
    set_lower_PML(-0.1)
    set_upper_PML(-0.1)
    t = 2 # thickness of each slab. If t>>wavelength solution approaches to planar. #0.2
   
    """    
    Its geometry is a fiber. But core layer will alternate in Fibonacci. sequence
    like a Fiber Bragg Grating
    """
    A = Slab(A_m(t))
    B = Slab(B_m(t))
    C = Slab(C_m(t))
    air = Slab(air_m(t))
    
    s = set_structure()
    
    #s.calc()
    #s.plot()

    # Loop over incidence wavelengths.
    for l in arange(min_l, max_l, dl):
        set_lambda(l) # new
        set_N(1) # sets the number of modes used in the series expansion.
        
        # Plane wave with amplitude I and angle theta (radians).
        II = 1.0
        theta = 0.0
        s.set_inc_field_plane_wave(II, theta*pi/180, eps)
        
        # set_lambda(l) old
        calc_R_vs_l()

    #fanimate_fields()    
    
### SOLUTION: circ    
elif sol=="circ":
    set_material()    
    # Define waveguide sections.
    set_circ_PML(-0.1)
    t = 0.5 # 0.5 (um) # thickness of each slab. #0.2

    A = Circ(clad_m(4*t) + A_m(t) + clad_m(4*t))
    B = Circ(clad_m(4*t) + B_m(t) + clad_m(4*t))
    C = Circ(clad_m(4*t) + C_m(t) + clad_m(4*t))
    air = Circ(air_m(9*t))
    
    """
    A = Circ(A_m(t))
    B = Circ(B_m(t))
    air = Circ(air_m(t))
    """    
    
    s = set_structure()
     
    # Loop over incidence wavelengths.
    for l in arange(min_l, max_l, dl):
        set_lambda(l) # new
        set_N(1) # sets the number of modes used in the series expansion.
        set_circ_order(0)
        
        calc_R_vs_l()

    #fanimate_fields()
    free_tmps()

plot_save()
print(filename + " calculation is finished.")