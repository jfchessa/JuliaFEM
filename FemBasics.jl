__precompile__()

module FemBasics

export REALTYPE, IDTYPE    

export shape_line2!, dshape_line2!, shape_line3!, dshape_line3!
export shape_tria3!, dshape_tria3!, shape_tria6!, dshape_tria6!
export shape_quad4!, dshape_quad4!, shape_quad8!, dshape_quad8!
export shape_tetra4!, dshape_tetra4!, shape_tetra10!, dshape_tetra10!
export shape_hexa8!, dshape_hexa8!, shape_hexa20!, dshape_hexa20!
export shape_penta6!, dshape_penta6!, shape_penta15!, dshape_penta15!
export shape_pyramid5!, dshape_pyramid5!

export GAUSS1D_1PT, GAUSS1D_1WT, GAUSS1D_2PT, GAUSS1D_2WT, GAUSS1D_3PT, GAUSS1D_3WT, 
       GAUSS1D_4PT, GAUSS1D_4WT, GAUSS1D_5PT, GAUSS1D_5WT, GAUSS1D_6PT, GAUSS1D_6WT, 
       GAUSS1D_7PT, GAUSS1D_7WT, GAUSS1D_8PT, GAUSS1D_8WT, GAUSS2D_1PT, GAUSS2D_1WT, 
       GAUSS2D_2PT, GAUSS2D_2WT, GAUSS2D_3PT, GAUSS2D_3WT, GAUSS2D_4PT, GAUSS2D_4WT, 
       GAUSS3D_1PT, GAUSS3D_1WT, GAUSS3D_2PT, GAUSS3D_2WT, GAUSS3D_3PT, GAUSS3D_3WT, 
       GAUSS3D_4PT, GAUSS3D_4WT, SIMPLEX2D_1PT, SIMPLEX2D_1WT, SIMPLEX2D_3PT, SIMPLEX2D_3WT, 
       SIMPLEX2D_7PT, SIMPLEX2D_7WT, SIMPLEX3D_1PT , SIMPLEX3D_1WT, SIMPLEX3D_4PT , SIMPLEX3D_4WT, 
       SIMPLEX3D_5PT , SIMPLEX3D_5WT
    
export gradshape1d!, gradshapeline2!, gradshape2d!, gradshapetria3!, gradshape3d!, gradshapetetra4!
export DelayedAssmMat, add_kmat!, assemble_mat
export setsctr!
export penaltybc!, fesolve!
export BTCBop!

using SparseArrays, LinearAlgebra
using LinearSolve
using IterativeSolvers

REALTYPE = Float64
IDTYPE = Int64

#-----------------------------------------------------------------------
# Basic Quadrature rules
# This has the basic quadrature rules defined as constants in the form
#
#       GAUSS%D_#PT, GAUSS%D_#WT, SIMPLEX%D_#PT, SIMPLEX%D_#WT
#
# where % is the spacial dimension of the rule and # is the number of 
# points in the rule.
include("./quaddata.jl")  

# -------------------------------------------------------------------------
#                     Basic element shape functions
#
#  Two node line element     1---------2
#                           xi=-1    xi=1
function shape_line2!(N, xi)
	"""
	shape_line2!(N, xi)
	"""
	N[1] = 0.5*( 1-xi )
	N[2] = 0.5*( 1+xi )
	return 2
end

function dshape_line2!(dNxi, xi)
	"""
	dshape_line2!(dNxi, xi)
	"""
	dNxi[1] = -0.5 
	dNxi[2] =  0.5
	return (2, 1)
end

#-----------------------------------------------------------------------
#
#  Three node line element     1----2----3
#                         xi= -1    0    1
#

function shape_line3!(N, xi)
	"""
	shape_line3!(N, xi)
	"""
	N[1] = -0.5*(1-xi)*xi
	N[2] = 1-xi^2 
	N[3] = 0.5*(1+xi)*xi 
	return 3
end

function dshape_line3!(dNxi, xi)
	dNxi[1] = xi-.5 
	dNxi[2] = -2*xi 
	dNxi[3] =  xi+.5 
	return (2,1)
end

#-----------------------------------------------------------------------
#
#                                     3
#  Constant strain triangle           !  \
#                                     !    \
#                                     1------2
#
#  Parent element node coordinates
#   node 1  xi = 0.0  eta = 0.0
#   node 2  xi = 1.0  eta = 0.0
#   node 3  xi = 0.0  eta = 1.0
#
function shape_tria3!(N, xi)
	"""
	shape_tria3!(N, xi)
	"""
	N[1] = 1-xi[1]-xi[2]
	N[2] = xi[1]
	N[3] = xi[2]
	return 3
end

function dshape_tria3!(dNxi, xi)
	"""
	dshape_tria3!(dNxi, xi)
	"""
    dNxi[1,1] = -1.0
    dNxi[2,1] =  1.0
    dNxi[3,1] =  0.0
    dNxi[1,2] = -1.0
    dNxi[2,2] =  0.0
    dNxi[3,2] =  1.0
	return (3,2)
end

#-----------------------------------------------------------------------
#
#                                    3
#  Linear strain triangle           !  \
#                                   6   5
#                                   !    \
#                                   1--4---2
#
#  Parent element node coordinates
#   node 1  xi = 0.0  eta = 0.0
#   node 2  xi = 1.0  eta = 0.0
#   node 3  xi = 0.0  eta = 1.0
#   node 4  xi = 0.5  eta = 0.0
#   node 5  xi = 0.5  eta = 0.5
#   node 6  xi = 0.0  eta = 0.5
#
function shape_tria6!(N, xi)
 	"""
 	shape_tria6!(N, xi)
	"""
	N[1] = 1-3*(xi[1]+xi[2])+4*xi[1]*xi[2]+2*(xi[1]*xi[1]+xi[2]*xi[2])
    N[2] = xi[1]*(2*xi[1]-1)
    N[3] = xi[2]*(2*xi[2]-1)
    N[4] = 4*xi[1]*(1-xi[1]-xi[2])
    N[5] = 4*xi[1]*xi[2]
    N[6] = 4*xi[2]*(1-xi[1]-xi[2])
	return 6
end

function dshape_tria6!(dNxi, coord)
	"""
	dshape_tria6!(N, xi)
	"""
	xi=coord[1]
	eta=coord[2]
	dNxi[1,1] =   4*(xi+eta)-3;  dNxi[1,2] =  4*(xi+eta)-3;
    dNxi[2,1] =         4*xi-1;  dNxi[2,2] =  0.0;
    dNxi[3,1] =            0.0;  dNxi[3,2] =  4*eta-1;
    dNxi[4,1] = 4*(1-eta-2*xi);  dNxi[4,2] = -4*xi;
    dNxi[5,1] =          4*eta;  dNxi[5,2] =  4*xi;
    dNxi[6,1] =         -4*eta;  dNxi[6,2] = 4*(1-xi-2*eta);
	return (6,2)
end

#-----------------------------------------------------------------------
#
#                                     4---------3
#  Four node quadrilateral element    !         !
#                                     !         !
#                                     1---------2
#
#  Parent element node coordinates
#   node 1  xi = -1.0  eta = -1.0
#   node 2  xi =  1.0  eta = -1.0
#   node 3  xi =  1.0  eta =  1.0
#   node 4  xi = -1.0  eta =  1.0
#
function shape_quad4!(N, xi)
	"""
	shape_quad4!(N, xi)
	"""
	s = xi[1]
	t = xi[2]
	N[1] = 0.25*(1-s)*(1-t)
	N[2] = 0.25*(1+s)*(1-t)
    N[3] = 0.25*(1+s)*(1+t)
    N[4] = 0.25*(1-s)*(1+t)
	return nl = 4
end

function dshape_quad4!(dNxi, xi)
	"""
	dshape_quad4!(dNxi, xi)
	"""
	s = xi[1]
	t = xi[2]
    dNxi[1, 1] = -0.25*(1-t);    dNxi[1, 2] = -0.25*(1-s);
    dNxi[2, 1] = 0.25*(1-t);     dNxi[2, 2] = -0.25*(1+s);
    dNxi[3, 1] = 0.25*(1+t);     dNxi[3, 2] =  0.25*(1+s);
    dNxi[4, 1] = -0.25*(1+t);    dNxi[4, 2] =  0.25*(1-s);
	return nl = (4,2)
end

#-----------------------------------------------------------------------
#
#                                     4----7----3
#  Eight node quadrilateral element   !         !
#                                     8         6 
#                                     !         !
#                                     1----5----2
#
#  Parent element node coordinates
#   node 1  xi = -1.0  eta = -1.0
#   node 2  xi =  1.0  eta = -1.0
#   node 3  xi =  1.0  eta =  1.0
#   node 4  xi = -1.0  eta =  1.0
#   node 5  xi =  0.0  eta = -1.0
#   node 6  xi =  1.0  eta =  0.0
#   node 7  xi =  0.0  eta =  1.0
#   node 8  xi = -1.0  eta =  0.0
#
function shape_quad8!(N, xi)
	"""
	shape_quad8!(N, xi)
	"""
	s = xi[1]
	t = xi[2]
    N[1] = 0.25*(-s^2*t - s*t^2 + s^2 + s*t + t^2 - 1)
    N[2] = 0.25*(-s^2*t + s*t^2 + s^2 - s*t + t^2 - 1)
    N[3] = 0.25*(s^2*t + s*t^2 + s^2 + s*t + t^2 - 1)
    N[4] = 0.25*(s^2*t - s*t^2 + s^2 - s*t + t^2 - 1)
    N[5] = 0.25*(2*s^2*t - 2*s^2 - 2*t + 2)
    N[6] = 0.25*(-2*s*t^2 - 2*t^2 + 2*s + 2)
    N[7] = 0.25*(-2*s^2*t - 2*s^2 + 2*t + 2)
    N[8] = 0.25*(2*s*t^2 - 2*t^2 - 2*s + 2)

    return 8
end

function dshape_quad8!(dN, xi)
	"""
	dshape_quad8!(dN, xi)
	"""
	s = xi[1]
	t = xi[2]
    dN[1,1] = 0.25*(-2*s*t - t^2 + 2*s + t)
    dN[1,2] = 0.25*(-s^2 - 2*s*t + s + 2*t)
    dN[2,1] = 0.25*(-2*s*t + t^2 + 2*s - t)
    dN[2,2] = 0.25*(-s^2 + 2*s*t - s + 2*t)
    dN[3,1] = 0.25*(2*s*t + t^2 + 2*s + t)
    dN[3,2] = 0.25*(s^2 + 2*s*t + s + 2*t)
    dN[4,1] = 0.25*(2*s*t - t^2 + 2*s - t)
    dN[4,2] = 0.25*(s^2 - 2*s*t - s + 2*t)
    dN[5,1] = s*t - s
    dN[5,2] = 0.5*(s^2 - 1)
    dN[6,1] = 0.5*(-t^2 + 1)
    dN[6,2] = -s*t - t
    dN[7,1] = -s*t - s
    dN[7,2] = 0.5*(-s^2 + 1)
    dN[8,1] = 0.5*(t^2 - 1)
    dN[8,2] = s*t - t

    return (8,2)
end

#-----------------------------------------------------------------------
#
#
#  Four node tetrahedral element
#
#  Parent element node coordinates
#   node 1  xi =  0.0  eta =  0.0  zeta =  0.0
#   node 2  xi =  1.0  eta =  0.0  zeta =  0.0
#   node 3  xi =  0.0  eta =  1.0  zeta =  0.0
#   node 4  xi =  0.0  eta =  0.0  zeta =  1.0
#
function shape_tetra4!(N, xi)
	"""
	shape_tetra4!(N, xi)
	"""
	N[1] = 1-xi[1]-xi[2]-xi[3]
    N[2] = xi[1] 
    N[3] = xi[2] 
    N[4] = xi[3] 
	return 4
end

function dshape_tetra4!(dN, xi)
	"""
	dshape_tetra4!(dNdxi, xi)
	"""
    dN[1,1] = -1.0; dN[1,2] = -1.0; dN[1,3] = -1.0;
    dN[2,1] =  1.0; dN[2,2] =  0.0; dN[2,3] =  0.0;
    dN[3,1] =  0.0; dN[3,2] =  1.0; dN[3,3] =  0.0;
    dN[4,1] =  0.0; dN[4,2] =  0.0; dN[4,3] =  1.0;

	return (4,3)
end

#-----------------------------------------------------------------------
#
#
#  Ten node tetrahedral element
#
#      nodes 1 2 3 4 are on the vertecies in the xi, eta, and zeta direcections
#      node 5 is between (1,2)
#      node 6 is between (2,3)
#      node 7 is between (3,1)
#      node 8 is between (1,4)
#      node 9 is between (2,4)
#      node 10 is between (3,4)
#
#  Parent element node coordinates
#   node  1  xi =  0.0  eta =  0.0  zeta =  0.0
#   node  2  xi =  1.0  eta =  0.0  zeta =  0.0
#   node  3  xi =  0.0  eta =  1.0  zeta =  0.0
#   node  4  xi =  0.0  eta =  0.0  zeta =  1.0
#   node  5  xi =  0.5  eta =  0.0  zeta =  0.0
#   node  6  xi =  0.5  eta =  0.5  zeta =  0.0
#   node  7  xi =  0.0  eta =  0.5  zeta =  0.0
#   node  8  xi =  0.0  eta =  0.0  zeta =  0.5
#   node  9  xi =  0.5  eta =  0.0  zeta =  0.5
#   node 10  xi =  0.0  eta =  0.5  zeta =  0.5
#
function shape_tetra10!(N, xi)
	"""
	shape_tetra10!(N, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];
	rr = r^2
	ss = s^2
	tt = t^2
	rs = r*s
	st = s*t
	rt = t*r
	N[1] = 1 - 3*r + 2*rr + 4*rs + 4*rt - 3*s + 2*ss + 4*st - 3*t + 2*tt
	N[2] = -r + 2*rr
	N[3] = -s + 2*ss
	N[4] = -t + 2*tt
	N[5] = 4*r - 4*rr - 4*rs - 4*rt
	N[6] = 4*rs
	N[7] = -4*rs + 4*s - 4*ss - 4*st
	N[8] = -4*rt - 4*st + 4*t - 4*tt
	N[9] = 4*rt
	N[10]= 4*st
	return 10
end

function dshape_tetra10!(dNxi, xi)
	"""
	dshape_tetra10!(dNxi, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

	dNxi[1,1] = -3 + 4*r + 4*s + 4*t
	dNxi[1,2] = dNxi[1,1]
	dNxi[1,3] = dNxi[1,1]

	dNxi[2,1] = -1 + 4*r
	dNxi[2,2] = 0.0
	dNxi[2,3] = 0.0

	dNxi[3,1] = 0.0
	dNxi[3,2] = -1 + 4*s
	dNxi[3,3] = 0.0

	dNxi[4,1] = 0.0
	dNxi[4,2] = 0.0
	dNxi[4,3] = -1 + 4*t

	dNxi[5,1] = 4 - 8*r - 4*s - 4*t
	dNxi[5,2] = -4*r
	dNxi[5,3] = -4*r

	dNxi[6,1] = 4*s
	dNxi[6,2] = 4*r
	dNxi[6,3] = 0.0

	dNxi[7,1] = -4*s
	dNxi[7,2] = 4 - 4*r - 8*s - 4*t
	dNxi[7,3] =-4*s

	dNxi[8,1] = -4*t
	dNxi[8,2] = -4*t
	dNxi[8,3] = 4 - 4*r - 4*s - 8*t

	dNxi[9,1] = 4*t
	dNxi[9,2] = 0.0
	dNxi[9,3] = 4*r

	dNxi[10,1] = 0.0
	dNxi[10,2] = 4*t
	dNxi[10,3] = 4*s

	return (10, 3)
end

#-----------------------------------------------------------------------
#
# Eight node hexhedral element
#
#  Parent element node coordinates
#   node  1  xi =  -1.0  eta =  -1.0  zeta =  -1.0
#   node  2  xi =   1.0  eta =  -1.0  zeta =  -1.0
#   node  3  xi =   1.0  eta =   1.0  zeta =  -1.0
#   node  4  xi =  -1.0  eta =   1.0  zeta =  -1.0
#   node  5  xi =  -1.0  eta =  -1.0  zeta =   1.0
#   node  6  xi =   1.0  eta =  -1.0  zeta =   1.0
#   node  7  xi =   1.0  eta =   1.0  zeta =   1.0
#   node  8  xi =  -1.0  eta =   1.0  zeta =   1.0
#
function shape_hexa8!(N, xi)
	"""
	shape_hexa8!(N, xi)
	"""
    I11 = 0.5 - 0.5*xi[1]
    I12 = 0.5 - 0.5*xi[2]
    I13 = 0.5 - 0.5*xi[3]
    I21 = 0.5 + 0.5*xi[1]
    I22 = 0.5 + 0.5*xi[2]
    I23 = 0.5 + 0.5*xi[3]

    N[1] = I11*I12*I13
    N[2] = I21*I12*I13
    N[3] = I21*I22*I13
    N[4] = I11*I22*I13
    N[5] = I11*I12*I23
    N[6] = I21*I12*I23
    N[7] = I21*I22*I23
    N[8] = I11*I22*I23

    return 8

end # shape_hexa8

function dshape_hexa8!(dNxi, coord) 
	"""
	dshape_hexa8!(dNdxi, xi)
	"""
	xi =   coord[1]
	eta =  coord[2]
	zeta = coord[3]

    dNxi[1,1] = 0.125*(-1+eta+zeta-eta*zeta)   
    dNxi[1,2] = 0.125*(-1+xi+zeta-xi*zeta) 
    dNxi[1,3] = 0.125*(-1+xi+eta-xi*eta)
    dNxi[2,1] = 0.125*(1-eta-zeta+eta*zeta)  
    dNxi[2,2] = 0.125*(-1-xi+zeta+xi*zeta) 
    dNxi[2,3] = 0.125*(-1-xi+eta+xi*eta)
    dNxi[3,1] = 0.125*(1+eta-zeta-eta*zeta)   
    dNxi[3,2] = 0.125*(1+xi-zeta-xi*zeta) 
    dNxi[3,3] = 0.125*(-1-xi-eta-xi*eta)
    dNxi[4,1] = 0.125*(-1-eta+zeta+eta*zeta)   
    dNxi[4,2] = 0.125*(1-xi-zeta+xi*zeta) 
    dNxi[4,3] = 0.125*(-1+xi-eta+xi*eta)     
    dNxi[5,1] = 0.125*(-1+eta-zeta+eta*zeta)  
    dNxi[5,2] = 0.125*(-1+xi-zeta+xi*zeta)  
    dNxi[5,3] = 0.125*(1-xi-eta+xi*eta)
    dNxi[6,1] = 0.125*(1-eta+zeta-eta*zeta)  
    dNxi[6,2] = 0.125*(-1-xi-zeta-xi*zeta)   
    dNxi[6,3] = 0.125*(1+xi-eta-xi*eta)
    dNxi[7,1] = 0.125*(1+eta+zeta+eta*zeta)    
    dNxi[7,2] = 0.125*(1+xi+zeta+xi*zeta)   
    dNxi[7,3] = 0.125*(1+xi+eta+xi*eta)
    dNxi[8,1] = 0.125*(-1-eta-zeta-eta*zeta)    
    dNxi[8,2] = 0.125*(1-xi+zeta-xi*zeta)   
    dNxi[8,3] = 0.125*(1-xi+eta-xi*eta)

	return (8,3)

end # shape_hexa8

#-----------------------------------------------------------------------
#
#   Twenty node hexhedral element
#
#   4----11---3  20--------19   8---15----7
#   !         !   !         !   !         !
#  12   t=-1  10  !   t=0   !  16   t=1   14 
#   !         !   !         !   !         !
#   1----9----2  17--------18   5---13----6
#
#  Parent element node coordinates
#   node  1  xi =  -1.0  eta =  -1.0  zeta =  -1.0
#   node  2  xi =   1.0  eta =  -1.0  zeta =  -1.0
#   node  3  xi =   1.0  eta =   1.0  zeta =  -1.0
#   node  4  xi =  -1.0  eta =   1.0  zeta =  -1.0
#   node  5  xi =  -1.0  eta =  -1.0  zeta =   1.0
#   node  6  xi =   1.0  eta =  -1.0  zeta =   1.0
#   node  7  xi =   1.0  eta =   1.0  zeta =   1.0
#   node  8  xi =  -1.0  eta =   1.0  zeta =   1.0
#   node  9  xi =   0.0  eta =  -1.0  zeta =  -1.0
#   node 10  xi =   1.0  eta =   0.0  zeta =  -1.0
#   node 11  xi =   0.0  eta =   1.0  zeta =  -1.0
#   node 12  xi =  -1.0  eta =   0.0  zeta =  -1.0
#   node 13  xi =   0.0  eta =  -1.0  zeta =   1.0
#   node 14  xi =   1.0  eta =   0.0  zeta =   1.0
#   node 15  xi =   0.0  eta =   1.0  zeta =   1.0
#   node 16  xi =  -1.0  eta =   0.0  zeta =   1.0
#   node 17  xi =  -1.0  eta =  -1.0  zeta =   0.0
#   node 18  xi =   1.0  eta =  -1.0  zeta =   0.0
#   node 19  xi =   1.0  eta =   1.0  zeta =   0.0
#   node 20  xi =  -1.0  eta =   1.0  zeta =   0.0
#
function shape_hexa20!(N, xi)
	"""
	shape_hexa20!(N, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

    N[1]  = -0.125*(1-r)*(1-s)*(1-t)*(2+r+s+t)
    N[2]  =  0.125*(1+r)*(1-s)*(1-t)*(r-s-t-2)
    N[3]  =  0.125*(1+r)*(1+s)*(1-t)*(r+s-t-2)
    N[4]  =  0.125*(1-r)*(s+1)*(1-t)*(-r+s-t-2)
    N[5]  =  0.125*(1-r)*(1-s)*(1+t)*(-r-s+t-2)
    N[6]  =  0.125*(1+r)*(1-s)*(1+t)*(r-s+t-2)
    N[7]  =  0.125*(1+r)*(1+s)*(1+t)*(r+s+t-2)
    N[8]  =  0.125*(1-r)*(1+s)*(1+t)*(-r+s+t-2)
    N[9]  =  0.25*(1-r^2)*(1-s)*(1-t)
    N[10] =  0.25*(1+r)*(1-s^2)*(1-t)
    N[11] =  0.25*(1-r^2)*(1+s)*(1-t)
    N[12] =  0.25*(1-r)*(1-s^2)*(1-t)
    N[13] =  0.25*(1-r^2)*(1-s)*(1+t)
    N[14] =  0.25*(1+r)*(1-s^2)*(1+t)
    N[15] =  0.25*(1-r^2)*(1+s)*(1+t)
    N[16] =  0.25*(1-r)*(1-s^2)*(1+t)
    N[17] =  0.25*(1-r)*(1-s)*(1-t^2)
    N[18] =  0.25*(1+r)*(1-s)*(1-t^2)
    N[19] =  0.25*(1+r)*(1+s)*(1-t^2)
    N[20] =  0.25*(1-r)*(1+s)*(1-t^2)
    
    return 20
end

function dshape_hexa20!(dN, xi)
	"""
	dshape_hexa20!(dNdxi, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

    dN[1,1] = 0.125*(r + s + t + 2)*(s - 1)*(t - 1) + 0.125*(r - 1)*(s - 1)*(t - 1)
    dN[1,2] = 0.125*(r + s + t + 2)*(r - 1)*(t - 1) + 0.125*(r - 1)*(s - 1)*(t - 1)
    dN[1,3] = 0.125*(r + s + t + 2)*(r - 1)*(s - 1) + 0.125*(r - 1)*(s - 1)*(t - 1)
    dN[2,1] = 0.125*(r - s - t - 2)*(s - 1)*(t - 1) + 0.125*(r + 1)*(s - 1)*(t - 1)
    dN[2,2] = 0.125*(r - s - t - 2)*(r + 1)*(t - 1) - 0.125*(r + 1)*(s - 1)*(t - 1)
    dN[2,3] = 0.125*(r - s - t - 2)*(r + 1)*(s - 1) - 0.125*(r + 1)*(s - 1)*(t - 1)
    dN[3,1] = -0.125*(r + s - t - 2)*(s + 1)*(t - 1) - 0.125*(r + 1)*(s + 1)*(t - 1)
    dN[3,2] = -0.125*(r + s - t - 2)*(r + 1)*(t - 1) - 0.125*(r + 1)*(s + 1)*(t - 1)
    dN[3,3] = -0.125*(r + s - t - 2)*(r + 1)*(s + 1) + 0.125*(r + 1)*(s + 1)*(t - 1)
    dN[4,1] = -0.125*(r - s + t + 2)*(s + 1)*(t - 1) - 0.125*(r - 1)*(s + 1)*(t - 1)
    dN[4,2] = -0.125*(r - s + t + 2)*(r - 1)*(t - 1) + 0.125*(r - 1)*(s + 1)*(t - 1)
    dN[4,3] = -0.125*(r - s + t + 2)*(r - 1)*(s + 1) - 0.125*(r - 1)*(s + 1)*(t - 1)
    dN[5,1] = -0.125*(r + s - t + 2)*(s - 1)*(t + 1) - 0.125*(r - 1)*(s - 1)*(t + 1)
    dN[5,2] = -0.125*(r + s - t + 2)*(r - 1)*(t + 1) - 0.125*(r - 1)*(s - 1)*(t + 1)
    dN[5,3] = -0.125*(r + s - t + 2)*(r - 1)*(s - 1) + 0.125*(r - 1)*(s - 1)*(t + 1)
    dN[6,1] = -0.125*(r - s + t - 2)*(s - 1)*(t + 1) - 0.125*(r + 1)*(s - 1)*(t + 1)
    dN[6,2] = -0.125*(r - s + t - 2)*(r + 1)*(t + 1) + 0.125*(r + 1)*(s - 1)*(t + 1)
    dN[6,3] = -0.125*(r - s + t - 2)*(r + 1)*(s - 1) - 0.125*(r + 1)*(s - 1)*(t + 1)
    dN[7,1] = 0.125*(r + s + t - 2)*(s + 1)*(t + 1) + 0.125*(r + 1)*(s + 1)*(t + 1)
    dN[7,2] = 0.125*(r + s + t - 2)*(r + 1)*(t + 1) + 0.125*(r + 1)*(s + 1)*(t + 1)
    dN[7,3] = 0.125*(r + s + t - 2)*(r + 1)*(s + 1) + 0.125*(r + 1)*(s + 1)*(t + 1)
    dN[8,1] = 0.125*(r - s - t + 2)*(s + 1)*(t + 1) + 0.125*(r - 1)*(s + 1)*(t + 1)
    dN[8,2] = 0.125*(r - s - t + 2)*(r - 1)*(t + 1) - 0.125*(r - 1)*(s + 1)*(t + 1)
    dN[8,3] = 0.125*(r - s - t + 2)*(r - 1)*(s + 1) - 0.125*(r - 1)*(s + 1)*(t + 1)
    dN[9,1] = -0.5*r*(s - 1)*(t - 1)
    dN[9,2] = -0.25*(r^2 - 1)*(t - 1)
    dN[9,3] = -0.25*(r^2 - 1)*(s - 1)
    dN[10,1] = 0.25*(s^2 - 1)*(t - 1)
    dN[10,2] = 0.5*(r + 1)*s*(t - 1)
    dN[10,3] = 0.25*(s^2 - 1)*(r + 1)
    dN[11,1] = 0.5*r*(s + 1)*(t - 1)
    dN[11,2] = 0.25*(r^2 - 1)*(t - 1)
    dN[11,3] = 0.25*(r^2 - 1)*(s + 1)
    dN[12,1] = -0.25*(s^2 - 1)*(t - 1)
    dN[12,2] = -0.5*(r - 1)*s*(t - 1)
    dN[12,3] = -0.25*(s^2 - 1)*(r - 1)
    dN[13,1] = 0.5*r*(s - 1)*(t + 1)
    dN[13,2] = 0.25*(r^2 - 1)*(t + 1)
    dN[13,3] = 0.25*(r^2 - 1)*(s - 1)
    dN[14,1] = -0.25*(s^2 - 1)*(t + 1)
    dN[14,2] = -0.5*(r + 1)*s*(t + 1)
    dN[14,3] = -0.25*(s^2 - 1)*(r + 1)
    dN[15,1] = -0.5*r*(s + 1)*(t + 1)
    dN[15,2] = -0.25*(r^2 - 1)*(t + 1)
    dN[15,3] = -0.25*(r^2 - 1)*(s + 1)
    dN[16,1] = 0.25*(s^2 - 1)*(t + 1)
    dN[16,2] = 0.5*(r - 1)*s*(t + 1)
    dN[16,3] = 0.25*(s^2 - 1)*(r - 1)
    dN[17,1] = -0.25*(t^2 - 1)*(s - 1)
    dN[17,2] = -0.25*(t^2 - 1)*(r - 1)
    dN[17,3] = -0.5*(r - 1)*(s - 1)*t
    dN[18,1] = 0.25*(t^2 - 1)*(s - 1)
    dN[18,2] = 0.25*(t^2 - 1)*(r + 1)
    dN[18,3] = 0.5*(r + 1)*(s - 1)*t
    dN[19,1] = -0.25*(t^2 - 1)*(s + 1)
    dN[19,2] = -0.25*(t^2 - 1)*(r + 1)
    dN[19,3] = -0.5*(r + 1)*(s + 1)*t
    dN[20,1] = 0.25*(t^2 - 1)*(s + 1)
    dN[20,2] = 0.25*(t^2 - 1)*(r - 1)
    dN[20,3] = 0.5*(r - 1)*(s + 1)*t

    return (20,3)
end

#
# Six node pentahedral element
#
# Tensor product of a CST with a [-1,1] Legendre basis in the zeta direction
#
#  Parent element node coordinates
#   node 1  xi = 0.0  eta = 0.0  zeta = -1.0
#   node 2  xi = 1.0  eta = 0.0  zeta = -1.0
#   node 3  xi = 0.0  eta = 1.0  zeta = -1.0
#   node 4  xi = 0.0  eta = 0.0  zeta =  1.0
#   node 5  xi = 1.0  eta = 0.0  zeta =  1.0
#   node 6  xi = 0.0  eta = 1.0  zeta =  1.0
#
function shape_penta6!(N, xi)
	"""
	shape_penta6!(N, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];
	N[1] = 0.5*(1 - r - s)*(1 + t)
	N[2] = 0.5*r*(1 - t)
	N[3] = 0.5*s*(1 - t)
	N[4] = 0.5*(1 - r - s)*(1 + t)
	N[5] = 0.5*r*(1 + t)
	N[6] = 0.5*s*(1 + t)
	nl = 6
end

function dshape_penta6!(dNxi, xi)
	"""
	dshape_penta6!(dNxi, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

	dNxi[1,1] = 0.5*(-1 - t)
	dNxi[1,2] = 0.5*(-1 - t)
	dNxi[1,3] = 0.5*(1 - r - s)

	dNxi[2,1] = 0.5*(1 - t)
	dNxi[2,2] = 0.0
	dNxi[2,3] = -0.5*r

	dNxi[3,1] = 0.0
	dNxi[3,2] = 0.5*(1 - t)
	dNxi[3,3] = -0.5*s

	dNxi[4,1] = 0.5*(-1 - t)
	dNxi[4,2] = 0.5*(-1 - t)
	dNxi[4,3] = 0.5*(1 - r - s)

	dNxi[5,1] = 0.5*(1 + t)
	dNxi[5,2] = 0.0
	dNxi[5,3] = 0.5*r

	dNxi[6,1] = 0.0
	dNxi[6,2] = 0.5*(1 + t)
	dNxi[6,3] = 0.5*s

	nl = (6,3)
end

#
# Fifteen node pentahedral element
#
# Tensor product of a CST with a [-1,1] Legendre basis in the zeta direction
#
#  Parent element node coordinates
#   node  1  xi = 0.0  eta = 0.0  zeta = -1.0
#   node  2  xi = 1.0  eta = 0.0  zeta = -1.0
#   node  3  xi = 0.0  eta = 1.0  zeta = -1.0
#   node  4  xi = 0.5  eta = 0.0  zeta = -1.0
#   node  5  xi = 0.5  eta = 0.5  zeta = -1.0
#   node  6  xi = 0.0  eta = 0.5  zeta = -1.0
#   node  7  xi = 0.0  eta = 0.0  zeta =  1.0
#   node  8  xi = 1.0  eta = 0.0  zeta =  1.0
#   node  9  xi = 0.0  eta = 1.0  zeta =  1.0
#   node 10  xi = 0.5  eta = 0.0  zeta =  1.0
#   node 11  xi = 0.5  eta = 0.5  zeta =  1.0
#   node 12  xi = 0.0  eta = 0.5  zeta =  1.0
#   node 13  xi = 0.0  eta = 0.0  zeta =  0.0
#   node 14  xi = 1.0  eta = 0.0  zeta =  0.0
#   node 15  xi = 0.0  eta = 1.0  zeta =  0.0
#

function shape_penta15!(N, xi)
	"""
	shape_penta15!(N, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];
    
    N[ 1 ] =  (2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*t*(0.5*t - 0.5)
    N[ 2 ] =  (2*r - 1)*r*t*(0.5*t - 0.5)
    N[ 3 ] =  (2*s - 1)*s*t*(0.5*t - 0.5)
    N[ 4 ] =  -4*(r + s - 1)*r*t*(0.5*t - 0.5)
    N[ 5 ] =  4*r*s*t*(0.5*t - 0.5)
    N[ 6 ] =  -4*(r + s - 1)*s*t*(0.5*t - 0.5)
    N[ 7 ] =  (2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*t*(0.5*t + 0.5)
    N[ 8 ] =  (2*r - 1)*r*t*(0.5*t + 0.5)
    N[ 9 ] =  (2*s - 1)*s*t*(0.5*t + 0.5)
    N[ 10 ] =  -4*(r + s - 1)*r*t*(0.5*t + 0.5)
    N[ 11 ] =  4*r*s*t*(0.5*t + 0.5)
    N[ 12 ] =  -4*(r + s - 1)*s*t*(0.5*t + 0.5)
    N[ 13 ] =  (r + s - 1)*(t + 1)*(t - 1)
    N[ 14 ] =  -r*(t + 1)*(t - 1)
    N[ 15 ] =  -s*(t + 1)*(t - 1)

	nl = 15
end

function dshape_penta15!(dN, xi)
	"""
	dshape_penta15!(dNxi, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

    dN[ 1 , 1 ] =  (4*r + 4*s - 3)*t*(0.5*t - 0.5)
    dN[ 2 , 1 ] =  (2*r - 1)*t*(0.5*t - 0.5) + 2*r*t*(0.5*t - 0.5)
    dN[ 3 , 1 ] =  0
    dN[ 4 , 1 ] =  -4*(r + s - 1)*t*(0.5*t - 0.5) - 4*r*t*(0.5*t - 0.5)
    dN[ 5 , 1 ] =  4*s*t*(0.5*t - 0.5)
    dN[ 6 , 1 ] =  -4*s*t*(0.5*t - 0.5)
    dN[ 7 , 1 ] =  (4*r + 4*s - 3)*t*(0.5*t + 0.5)
    dN[ 8 , 1 ] =  (2*r - 1)*t*(0.5*t + 0.5) + 2*r*t*(0.5*t + 0.5)
    dN[ 9 , 1 ] =  0
    dN[ 10 , 1 ] =  -4*(r + s - 1)*t*(0.5*t + 0.5) - 4*r*t*(0.5*t + 0.5)
    dN[ 11 , 1 ] =  4*s*t*(0.5*t + 0.5)
    dN[ 12 , 1 ] =  -4*s*t*(0.5*t + 0.5)
    dN[ 13 , 1 ] =  (t + 1)*(t - 1)
    dN[ 14 , 1 ] =  -(t + 1)*(t - 1)
    dN[ 15 , 1 ] =  0

    dN[ 1 , 2 ] =  (4*r + 4*s - 3)*t*(0.5*t - 0.5)
    dN[ 2 , 2 ] =  0
    dN[ 3 , 2 ] =  (2*s - 1)*t*(0.5*t - 0.5) + 2*s*t*(0.5*t - 0.5)
    dN[ 4 , 2 ] =  -4*r*t*(0.5*t - 0.5)
    dN[ 5 , 2 ] =  4*r*t*(0.5*t - 0.5)
    dN[ 6 , 2 ] =  -4*(r + s - 1)*t*(0.5*t - 0.5) - 4*s*t*(0.5*t - 0.5)
    dN[ 7 , 2 ] =  (4*r + 4*s - 3)*t*(0.5*t + 0.5)
    dN[ 8 , 2 ] =  0
    dN[ 9 , 2 ] =  (2*s - 1)*t*(0.5*t + 0.5) + 2*s*t*(0.5*t + 0.5)
    dN[ 10 , 2 ] =  -4*r*t*(0.5*t + 0.5)
    dN[ 11 , 2 ] =  4*r*t*(0.5*t + 0.5)
    dN[ 12 , 2 ] =  -4*(r + s - 1)*t*(0.5*t + 0.5) - 4*s*t*(0.5*t + 0.5)
    dN[ 13 , 2 ] =  (t + 1)*(t - 1)
    dN[ 14 , 2 ] =  0
    dN[ 15 , 2 ] =  -(t + 1)*(t - 1)

    dN[ 1 , 3 ] =  0.5*(2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*t + (2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*(0.5*t - 0.5)
    dN[ 2 , 3 ] =  0.5*(2*r - 1)*r*t + (2*r - 1)*r*(0.5*t - 0.5)
    dN[ 3 , 3 ] =  0.5*(2*s - 1)*s*t + (2*s - 1)*s*(0.5*t - 0.5)
    dN[ 4 , 3 ] =  -2.0*(r + s - 1)*r*t - 4*(r + s - 1)*r*(0.5*t - 0.5)
    dN[ 5 , 3 ] =  2.0*r*s*t + 4*r*s*(0.5*t - 0.5)
    dN[ 6 , 3 ] =  -2.0*(r + s - 1)*s*t - 4*(r + s - 1)*s*(0.5*t - 0.5)
    dN[ 7 , 3 ] =  0.5*(2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*t + (2*r^2 + 4*r*s + 2*s^2 - 3*r - 3*s + 1)*(0.5*t + 0.5)
    dN[ 8 , 3 ] =  0.5*(2*r - 1)*r*t + (2*r - 1)*r*(0.5*t + 0.5)
    dN[ 9 , 3 ] =  0.5*(2*s - 1)*s*t + (2*s - 1)*s*(0.5*t + 0.5)
    dN[ 10 , 3 ] =  -2.0*(r + s - 1)*r*t - 4*(r + s - 1)*r*(0.5*t + 0.5)
    dN[ 11 , 3 ] =  2.0*r*s*t + 4*r*s*(0.5*t + 0.5)
    dN[ 12 , 3 ] =  -2.0*(r + s - 1)*s*t - 4*(r + s - 1)*s*(0.5*t + 0.5)
    dN[ 13 , 3 ] =  (r + s - 1)*(t + 1) + (r + s - 1)*(t - 1)
    dN[ 14 , 3 ] =  -r*(t + 1) - r*(t - 1)
    dN[ 15 , 3 ] =  -s*(t + 1) - s*(t - 1)

	nl = (15,3)
end

#------------------------------------------------------------------------------
#
# Five node pyramid element
#
#  Parent element node coordinates
#   node  1  xi = -1.0  eta = -1.0  zeta = 0.0
#   node  2  xi =  1.0  eta = -1.0  zeta = 0.0
#   node  3  xi =  1.0  eta =  1.0  zeta = 0.0
#   node  4  xi = -1.0  eta =  1.0  zeta = 0.0
#   node  5  xi =  0.0  eta =  0.0  zeta = 1.0
#

function shape_pyramid5!(N, xi)
	"""
	shape_pyramid5!(N, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];
    
    N[ 1 ] = 0.25*( r*s - r - s - t + 1 )
    N[ 2 ] = 0.25*( -r*s + r - s - t + 1 )
    N[ 3 ] = 0.25*( r*s + r + s - t + 1 )
    N[ 4 ] = 0.25*( -r*s - r + s - t + 1 )
    N[ 5 ] = t

	nl = 5
end

function dshape_pyramid5!(dN, xi)
	"""
	dshape_pyramid5!(dNxi, xi)
	"""
	r = xi[1]; s = xi[2]; t = xi[3];

    dN[ 1 , 1 ] = 0.25*( s - 1 )
    dN[ 1 , 2 ] = 0.25*( r - 1 )
    dN[ 1 , 3 ] = 0.25*( -1 )
    dN[ 2 , 1 ] = 0.25*( -s + 1 )
    dN[ 2 , 2 ] = 0.25*( -r - 1 )
    dN[ 2 , 3 ] = 0.25*( -1 )
    dN[ 3 , 1 ] = 0.25*( s + 1 )
    dN[ 3 , 2 ] = 0.25*( r + 1 )
    dN[ 3 , 3 ] = 0.25*( -1 )
    dN[ 4 , 1 ] = 0.25*( -s - 1 )
    dN[ 4 , 2 ] = 0.25*( -r + 1 )
    dN[ 4 , 3 ] = 0.25*( -1 )
    dN[ 5 , 1 ] = 0.0
    dN[ 5 , 2 ] = 0.0
    dN[ 5 , 3 ] = 1.0

	nl = (5,3)
end

#-----------------------------------------------------------------------------
# Isoparametric gradient functions
function gradshape1d!(dN, coord, nn::Int=size(coord)[2])
    """
    gradshape1d!(dN, coord, nn::Int=size(coord)[2])
    
     Computes the gradient of a 1D isoparametric element and returns the 
     determinant of the element Jacobian matrix

     dN - on input holds the gradient of the element shape function with
          resepect to the parent coordianate system, and on return holds
          the values of the gradient of the shape function with resepct 
          to the spacial coordinates
     coord - a matrix of the element nodeal coordinates. The nodal coordinates
            are stored in the columns of coord.
     nn::Int - the number of nodes in the element (default is the number of 
              columns in coord)
    """
    jac = 0.0
    for i in 1:n
        jac += coord[i]*dN[i]
    end
    invj = 1/jac
    dN .= dN*invj
    return jac
end

function gradshapeline2!(dN, coord)
    """
    gradshapeline2!(dN, coord)

    Computes the gradient of a Line2 element and returns the 
    determinant of the element Jacobian matrix (1/2 the length of the elemnt).  
    This is a bit faster than using gradshap1d! (probably not though).

    dN - On return holds the values of the gradient of the shape function with 
         resepct to the spacial coordinates (no need to input dNdxi)
    coord - a matrix of the element nodeal coordinates. The nodal coordinates
           are stored in the columns of coord.
    """
    jac = 0.5*abs(coord[2]-coord[1])
    dN .= dN*(1/jac)
    return jac
end

function gradshape2d!(dN, coord, nn::Int=size(coord)[2])
    """
     gradshape2d!(dN, coord, nn::Int)

     Computes the gradient of a 2D isoparametric element and returns the 
     determinant of the element Jacobian matrix

     dN - on input holds the gradient of the element shape function with
          resepect to the parent coordianate system, and on return holds
          the values of the gradient of the shape function with resepct 
          to the spacial coordinates
     coord - a matrix of the element nodeal coordinates. The nodal coordinates
            are stored in the columns of coord.
     nn::Int - the number of nodes in the element (default is the number of 
              columns in coord)

    """
    j11 = coord[1,1]*dN[1,1]
    j12 = coord[1,1]*dN[1,2]
    j21 = coord[2,1]*dN[1,1]
    j22 = coord[2,1]*dN[1,2]
    for k in 2:nn
        j11 += coord[1,k]*dN[k,1]
        j12 += coord[1,k]*dN[k,2]
        j21 += coord[2,k]*dN[k,1]
        j22 += coord[2,k]*dN[k,2]
    end
    jac = (j11*j22-j12*j21)
    invj = 1/jac
    for i in 1:nn
        dNi1 = dN[i,1]
        dNi2 = dN[i,2] 
        dN[i,1] = invj*(  dNi1*j22 - dNi2*j21 )
        dN[i,2] = invj*( -dNi1*j12 + dNi2*j11 )
    end
    return jac
end

function gradshapetria3!(dN, coord)
    """
     gradshapetria3!(dN, coord)

     Computes the gradient of a Tria3 element and returns the 
     determinant of the element Jacobian matrix.  This is a bit faster
     than using gradshap2d!.

     dN - On return holds the values of the gradient of the shape function with 
          resepct to the spacial coordinates (no need to input dNdxi)
     coord - a matrix of the element nodeal coordinates. The nodal coordinates
            are stored in the columns of coord.

    """
    x1 = coord[1,1]; x2 = coord[1,2]; x3 = coord[1,3];
    y1 = coord[2,1]; y2 = coord[2,2]; y3 = coord[2,3];
    jac = -((x2 - x3)*y1 - x1*(y2 - y3) + x3*y2 - x2*y3)
    invj = 1/jac
    dN[1,1] = (  y2 - y3 )*invj
    dN[1,2] = ( -x2 + x3 )*invj
    dN[2,1] = ( -y1 + y3 )*invj
    dN[2,2] = (  x1 - x3 )*invj
    dN[3,1] = (  y1 - y2 )*invj
    dN[3,2] = ( -x1 + x2 )*invj
    return jac
end

# This is globally allocated space for doing local calculations
# I know this is not a great idea, but does really speed things up
# Use views to help address this memory in the functions. 
const global _ScRaTcH_REAL_FEMBASICS_  = zeros(REALTYPE,3,3)
function gradshape3d!(dN, coord, nn=size(coord)[2])
    """
     gradshape3d!(dN, coord, nn::Int)

     Computes the gradient of a 3D isoparametric element and returns the 
     determinant of the element Jacobian matrix

     dN - on input holds the gradient of the element shape function with
          resepect to the parent coordianate system, and on return holds
          the values of the gradient of the shape function with resepct 
          to the spacial coordinates
     coord - a matrix of the element nodeal coordinates. The nodal coordinates
            are stored in the columns of coord.
     nn::Int - the number of nodes in the element (default is the number of 
              columns in coord, but not really used.)

    """
    jmat = @view _ScRaTcH_REAL_FEMBASICS_[1:3,1:3]    # THIS IS NOT OPTIMIZED
    jmat = coord*dN                                   # BUT NOT SURE WHAT TO DO
    dN .= dN/jmat
    return det(jmat)
end

function gradshapetetra4!(dN, coord)
    """
     gradshapetetra4!(dN, coord)

     Computes the gradient of a Tettra4 element and returns the 
     determinant of the element Jacobian matrix.  This is a bit faster
     than using gradshap3d!.

     dN - On return holds the values of the gradient of the shape function with 
          resepct to the spacial coordinates (no need to input dNdxi)
     coord - a matrix of the element nodeal coordinates. The nodal coordinates
            are stored in the columns of coord.

    """
    x1 = coord[1,1]; x2 = coord[1,2]; x3 = coord[1,3]; x4 = coord[1,4];
    y1 = coord[2,1]; y2 = coord[2,2]; y3 = coord[2,3]; y4 = coord[2,4];
    z1 = coord[3,1]; z2 = coord[3,2]; z3 = coord[3,3]; z4 = coord[3,4];
    jac = x3*y2*z1 - x4*y2*z1 - x2*y3*z1 + x4*y3*z1 + x2*y4*z1 - x3*y4*z1 - x3*y1*z2 + 
        x4*y1*z2 + x1*y3*z2 - x4*y3*z2 - x1*y4*z2 + x3*y4*z2 + x2*y1*z3 - x4*y1*z3 - 
        x1*y2*z3 + x4*y2*z3 + x1*y4*z3 - x2*y4*z3 - x2*y1*z4 + x3*y1*z4 + x1*y2*z4 - 
        x3*y2*z4 - x1*y3*z4 + x2*y3*z4
    invj = 1/jac
    dN[1, 1] = ( (y3 - y4)*z2 - y2*(z3 - z4) + y4*z3 - y3*z4 )*invj
    dN[1, 2] = ( -(x3 - x4)*z2 + x2*(z3 - z4) - x4*z3 + x3*z4 )*invj
    dN[1, 3] = ( (x3 - x4)*y2 - x2*(y3 - y4) + x4*y3 - x3*y4 )*invj
    dN[2, 1] = ( -(y3 - y4)*z1 + y1*(z3 - z4) - y4*z3 + y3*z4 )*invj
    dN[2, 2] = ( (x3 - x4)*z1 - x1*(z3 - z4) + x4*z3 - x3*z4 )*invj
    dN[2, 3] = ( -(x3 - x4)*y1 + x1*(y3 - y4) - x4*y3 + x3*y4 )*invj
    dN[3, 1] = ( (y2 - y4)*z1 - y1*(z2 - z4) + y4*z2 - y2*z4 )*invj
    dN[3, 2] = ( -(x2 - x4)*z1 + x1*(z2 - z4) - x4*z2 + x2*z4 )*invj
    dN[3, 3] = ( (x2 - x4)*y1 - x1*(y2 - y4) + x4*y2 - x2*y4 )*invj
    dN[4, 1] = ( -(y2 - y3)*z1 + y1*(z2 - z3) - y3*z2 + y2*z3 )*invj
    dN[4, 2] = ( (x2 - x3)*z1 - x1*(z2 - z3) + x3*z2 - x2*z3 )*invj
    dN[4, 3] = ( -(x2 - x3)*y1 + x1*(y2 - y3) - x3*y2 + x2*y3 )*invj
    return jac
end

#-----------------------------------------------------------------------------
struct DelayedAssmMat
    """
    DelayedAssmMat
    A structure that holds the element matrices and the scatter indices for a 
    group of elements.  This allows for the construction of the sparse matrix 
    to be delayed till after the calculation of all the element matrices.

    is -
    js- 
    kvals -

    """
    is::Array{IDTYPE,3}
    js::Array{IDTYPE,3}
    kvals::Array{REALTYPE,3}
end

DelayedAssmMat(ne::Int, kdim::Int) = DelayedAssmMat(Array{IDTYPE}(undef,kdim,kdim,ne), 
                 Array{IDTYPE}(undef,kdim,kdim,ne), Array{REALTYPE}(undef,kdim,kdim,ne)) 

function add_kmat!(K::DelayedAssmMat, e, ke, sctr)
"""
    function add_kmat!(K::DelayedAssmMat, e, ke, sctr)
        Adds an element matrix to a DelayedAssmMat structure
    K - 
    e - 
    ke - 
    sctr -
"""
    K.kvals[:,:,e] = ke   
    for (ii, Ii) in enumerate(sctr)
        for (jj, Jj) in enumerate(sctr)
            K.is[ii,jj,e] = Ii
            K.js[ii,jj,e] = Jj 
        end
    end
end

function assemble_mat(K::DelayedAssmMat)
    return(sparse(K.is[:], K.js[:], K.kvals[:]))
end
#-----------------------------------------------------------------------------
function setsctr!(sctr, conn, ndofpn=6, nn=length(conn))
    """
    function setsctr(sctr, conn, ndofpn, nn)

      sets the array sctr to be a finite element
      type scatter vector.  The element connectivity
      is given in conn, and the number of dof per
      node is given as ndofpn.
    """
    i = 1
    for n = conn[1:nn]
        Ii = ndofpn * (n - 1) + 1
        for s = 1:ndofpn
            sctr[i] = Ii
            i += 1
            Ii += 1
        end
    end
    ns = nn * ndofpn
end

BTCBop!(ke, B, C, a, add=true) = (add ? ke .+= B'*C*B*a : ke .= B'*C*B*a)
function BTCBop!(ke, B, m, n, C, a, add=true)
"""
function BTCBop!(ke, B, m, n, C, a, add=true)
        Computes ke += B^T*C*B*a 
    B is of dimensions m x n (ergo ke is n x n and C is m x m)
	If add = false else ke is initially zeroed and 
        ke = B^T*C*B*a 
"""
    if !add
		fill!(ke, 0.0)
	end
    Ckla = 0.0
    Blj = 0.0
    for l=1:m
        for k=1:m
            Ckla = C[k,l]*a
            for j=1:n
                Blj = B[l,j]
                for i=1:n
                    ke[i,j] += B[k,i]*Ckla*Blj
                end
            end
        end
    end
end

function penaltybc!(K, f, ifix, ival, penal::Real=10.0e5)
	"""
	function penaltybc!(K, f, ifix, ival=zeros(length(ifix)), penal=10.0e5)

	Enforces an essential BC on a Kd=f system using a penalty method
	"""
	for (ii, I) in enumerate(ifix)
		KII = sum(abs.(K[I,:]))*penal
		K[I,I] = KII
		f[I] = KII*ival[ii]
	end
end

function penaltybc!(K, f, ifix, penal::Real=10.0e5)
	"""
	function penaltybc!(K, f, ifix, penal=10.0e5)

	Enforces a zero essential BC on a Kd=f system using a penalty method
	"""
	for (ii, I) in enumerate(ifix)
		KII = sum(abs.(K[I,:]))*penal
		K[I,I] = KII
		f[I] = 0.0
	end
end

function enforcebc!(K, f, ifix, ival=nothing)
	"""
	function enforcebc!(K, f, ifix, ival=nothing)

	Enforces an essential BC on a Kd=f system using an equation swap.  
    This works for both trival and non-trivial essential BCs.
	"""
    m, n = size(K)
    if !isnothing(ival) 
        for (ii, I) in enumerate(ifix) 
            f .-= K[:,I]*ival[ii]
        end
    end
	for I in ifix
		KII = K[I,I]
		K[I,:] .= 0.0
		K[:,I] .= 0.0
        K[I,I] = KII
	end
    if isnothing(ival)
        f[ifix] .= 0.0
    else
        for (ii, I) in enumerate(ifix) 
            f[I] = K[I,I]*ival[ii]
        end
    end
end

function fesolve!(d, K::DelayedAssmMat, f, ifix, ival=nothing)
	"""
	fesolve(K, f, ifix, ival=zeros) 
	"""
    Kmat = assemble_mat(K)
    return  fesolve!(d, Kmat, f, ifix, ival)
end # of fesolve function

function fesolve!(d, K::AbstractSparseArray, f, ifix, ival=nothing, ebcmethod=enforcebc!)
	"""
	fesolve(K, f, ifix, ival=zeros, ebcmethod=enforcebc!) for sparse arrays
	"""
	Kr = K[ifix[:],:]
    if isnothing(ival)
        #penaltybc!(K, f, ifix)
        ebcmethod(K, f, ifix)
    else
	    #penaltybc!(K, f, ifix, ival)
        ebcmethod(K, f, ifix, ival)
    end
    d .= K \ f
	f[ifix[:],:] .= Kr*d
end # of fesolve function

end 
# ---------------------  END of FemBasics MODULE ----------------------#