__precompile__()

#-------------------------------------------------------------------------
#                         F E M    M O D U L E
#-------------------------------------------------------------------------
module FEM

export REALTYPE, IDTYPE

export shape_line2!, shape_line3!, shape_tria3!, shape_tria6!,
         shape_quad4!, shape_tetra4!, shape_tetra10!, shape_hexa8!,
		 shape_penta6!

export dshape_line2!, dshape_line3!, dshape_tria3!, dshape_tria6!,
		 dshape_quad4!, dshape_tetra4!, dshape_tetra10!, dshape_hexa8!,
		 dshape_penta6!

export qrule1d, qrulecomp, qruletria, qruletetra, qrulepenta, quadrule

export AbstractMaterial, GenMat
export AbstractProperty, GenProp, getprop, getmatprop
export AbstractElement, ElementData, QuadratureData
export numelem, elemnne, elemtopo, elemedim, elemsdim, elemvdim, elemorder,
       elemqtype, elemqrule, elemshape!, elemdshape!

export Line2D1, Line3D1
export Line2D2, Line3D2, Tria3D2, Tria6D2, Quad4D2
export Line2D3, Line3D3, Tria3D3, Tria6D3, Quad4D3,
       Tetra4D3, Tetra10D3, Hexa8D3, Penta6D3

export BTBop!, BBTop!, BTCBop!, BCBTop! 
export setsctr!, isojac!, gradbasis!, formbmat!, formnv!, formnvt!, fesolve!

export AbstractDofMap, FixedDofMap, DofMap, numDof, GDOF, addDof!, setsctr!

export DelayedAssmMat, add_kmat!, assemble_mat

export AbstractBC, GeneralBC, addDof!
#-------------------------------------------------------------------------
using SparseArrays, LinearSolve, LinearAlgebra

import Base.getindex, Base.setindex!, Base.length, Base.size

const THIRD = 0.333333333333333333333333333333333333333333333
const SIXTH = 0.166666666666666666666666666666666666666666667
const EIGTH = 0.125

REALTYPE = Float32
IDTYPE = Int32
# -------------------------------------------------------------------------
#                     Basic element shape functions
#
#  Two node line element     1---------2
#
function shape_line2!(N, xi::AbstractFloat=0.0)
	"""
	shape_line2!(N, xi::AbstractFloat=0.0)
	"""
	N[1:2] = 0.5*[ 1-xi 1+xi ]
	return 2
end

function shape_line2!(N, xi::AbstractArray{<:AbstractFloat,1})
	"""
	shape_line2!(N, xi::AbstractArray{<:AbstractFloat,1})
	"""
	nq = length(xi)
	for i=1:nq
		shape_line2!(view(N, 1:2, i), xi[i])
	end
	return (2, nq)
end

function shape_line2!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_line2!(N, xi)
	"""
	for i=1:size(xi,2)
		shape_line2!(view(N, 1:2, i), xi[1,i])
	end
	return (2, size(xi,2))
end

function dshape_line2!(dNxi, xi::AbstractFloat=0.0)
	"""
	dshape_line2!(dNxi, xi)
	"""
	dNxi[1:2, 1] = [-0.5 0.5]
	return nl = (2, 1)
end

function dshape_line2!(dNxi, xi::AbstractArray{<:AbstractFloat,1})
	"""
	dshape_line2!(dNxi, xi)
	"""
	for i=1:length(xi)
		dshape_line2!(view(dNxi, 1:2, 1, i), xi[i])
	end
	return nl = (2, 1, length(xi))
end

function dshape_line2!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_line2!(dNxi, xi)
	"""
	nq = size(xi,2)
	for i=1:nq
		dshape_line2!(view(dNxi, 1:2, 1, i), xi[1,i])
	end
	return nl = (2, 1, nq)
end

#
#  Three node line element     1----2----3
#

function shape_line3!(N, xi::AbstractFloat=0.0)
	"""
	shape_line3!(N, xi=0.0)
	"""
	N[1:3] = [ -0.5*(1-xi)*xi 1-xi^2 0.5*(1+xi)*xi ]
	nl = 3
end

function shape_line3!(N, xi::AbstractArray{<:AbstractFloat,1})
	"""
	shape_line3!(N, xi::AbstractArray{<:AbstractFloat,1})
	"""
	for i=1:length(xi)
		shape_line3!(view(N, 1:3, i), xi[i])
	end
	return (3, length(xi))
end

function shape_line3!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_line3!(N, xi)
	"""
	for i=1:size(xi,2)
		shape_line3!(view(N, 1:3, i), xi[1,i])
	end
	return (3, size(xi,2))
end

function dshape_line3!(dNxi, xi::AbstractFloat=0.0)
	dNxi[1:3, 1] = [ xi-.5 -2*xi xi+.5 ]
	nl = (2,1)
end

function dshape_line3!(dNxi, xi::AbstractArray{<:AbstractFloat,1})
	"""
	dshape_line3!(dNxi, xi)
	"""
	for i=1:length(xi)
		dshape_line3!(view(dNxi, 1:3, 1, i), xi[i])
	end
	return nl = (3, 1, length(xi))
end

function dshape_line3!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_line3!(dNxi, xi)
	"""
	nq = size(xi,2)
	for i=1:nq
		dshape_line3!(view(dNxi, 1:3, 1, i), xi[1,i])
	end
	return nl = (3, 1, nq)
end

#                                     3
#  Constant strain triangle           !  \
#                                     !    \
#                                     1------2

function shape_tria3!(N, xi::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
	"""
	shape_tria3!(N, xi)
	"""
	N[1] = 1-xi[1]-xi[2]
	N[2] = xi[1]
	N[3] = xi[2]
	nl = 3
end

function shape_tria3!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_tria3!(N, xi)
	"""
	nq = size(xi,2)
	for i=1:nq
		shape_tria3!(view(N, 1:3, i), view(xi, 1:2, i))
	end
	return (3, nq)
end

function dshape_tria3!(dNxi, xi::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
	"""
	dshape_tria3!(dNxi, xi)
	"""
	dNxi[1:3, 1:2] = [ -1 -1; 1 0; 0 1]
	nl = (3,2)
end

function dshape_tria3!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_tria3!(dNxi, xi)
	"""
	nq = size(xi,2)
	for i=1:nq
		dshape_tria3!(view(dNxi, 1:3, 1:2, i), view(xi, 1:2, i))
	end
	return nl = (3, 2, nq)
end

#                                    3
#  Linear strain triangle           !  \
#                                   6   5
#                                   !    \
#                                   1--4---2

function shape_tria6!(N, xi::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
 	"""
 	shape_tria6!(N, xi)
	"""
	N[1] = 1-3*(xi[1]+xi[2])+4*xi[1]*xi[2]+2*(xi[1]*xi[1]+xi[2]*xi[2])
    N[2] = xi[1]*(2*xi[1]-1)
    N[3] = xi[2]*(2*xi[2]-1)
    N[4] = 4*xi[1]*(1-xi[1]-xi[2])
    N[5] = 4*xi[1]*xi[2]
    N[6] = 4*xi[2]*(1-xi[1]-xi[2])
	nl = 6
end

function shape_tria6!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_tria6!(N, xi)
	"""
	nq = size(xi,2)
	nne = 6
	ed = 2
	for i=1:nq
		shape_tria6!(view(N, 1:nne, i), view(xi, 1:ed, i))
	end
	return (nne, nq)
end

function dshape_tria6!(dNxi, coord::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
	"""
	dshape_tria6!(N, xi)
	"""
	xi=coord[1]
	eta=coord[2]
	dNxi[1:6, 1:2] = [4*(xi+eta)-3   4*(xi+eta)-3;
                          4*xi-1              0;
                               0        4*eta-1;
                  4*(1-eta-2*xi)          -4*xi;
                           4*eta           4*xi;
                          -4*eta  4*(1-xi-2*eta) ]
	nl = (6,2)
end

function dshape_tria6!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_tria6!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 6
	ed = 2
	for i=1:nq
		dshape_tria6!(view(dNxi, 1:nne, 1:ed, i), view(xi, 1:ed, i))
	end
	return nl = (nne, ed, nq)
end


#                                     4---------3
#  Four node quadrilateral element    !         !
#                                     !         !
#                                     1---------2
function shape_quad4!(N, xi::AbstractArray{<:AbstractFloat,1}=zeros(2))
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

function shape_quad4!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_quad4!(dNxi, xi)
	"""
	for i=1:size(xi,2)
		shape_quad4!(view(N, 1:4, i), view(xi, 1:2, i))
	end
	return (2, size(xi,2))
end

function dshape_quad4!(dNxi, xi::AbstractArray{<:AbstractFloat,1}=zeros(2))
	"""
	dshape_quad4!(dNxi, xi)
	"""
	s = xi[1]
	t = xi[2]
    dNxi[1:4, 1:2] = 0.25*[-(1-t)    -(1-s);
		                     1-t     -(1+s);
		                     1+t       1+s;
                           -(1+t)      1-s]
	return nl = (4,2)
end

function dshape_quad4!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_quad4!(dNxi, xi)
	"""
	nq = size(xi,2)
	for i=1:nq
		dshape_quad4!(view(dNxi, 1:4, 1:2, i), view(xi, 1:2, i))
	end
	return nl = (4, 2, nq)
end

#
#  Four node tetrahedral element
#
#
function shape_tetra4!(N, xi::AbstractArray{<:AbstractFloat,1}=SIXTH*ones(3))
	"""
	shape_tetra4!(N, xi)
	"""
	N[1:4]=[ 1-xi[1]-xi[2]-xi[3] xi[1] xi[2] xi[3] ]
	nl = 4
end

function shape_tetra4!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_tetra4!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 4
	ed = 3
	for i=1:nq
		shape_tetra4!(view(N, 1:nne, i), view(xi, 1:ed, i))
	end
	return (nne, nq)
end

function dshape_tetra4!(dNxi, xi::AbstractArray{<:AbstractFloat,1}=SIXTH*ones(3))
	"""
	dshape_tetra4!(dNdxi, xi)
	"""
	dNxi[1:4,1:3] =  [ -1 -1 -1; 1 0 0; 0 1 0; 0 0 1 ]
	nl = (4,3)
end

function dshape_tetra4!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_tetra4!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 4
	ed = 3
	for i=1:nq
		dshape_tetra4!(view(dNxi, 1:nne, 1:ed, i), view(xi, 1:ed, i))
	end
	return nl = (nne, ed, nq)
end

#
#  Ten node tetrahedral element
#
#      nodes 1 2 3 4 are on the vertecies in the r, s, and t direcections
#      node 5 is between (1,2)
#      node 6 is between (2,3)
#      node 7 is between (3,1)
#      node 8 is between (1,4)
#      node 9 is between (2,4)
#      node 10 is between (3,4)
function shape_tetra10!(N, xi::AbstractArray{<:AbstractFloat,1}=SIXTH*ones(3))
	"""
	shape_tetra10!(N, xi)
	"""
	(r,s,t) = xi
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
	return nl = 10
end

function shape_tetra10!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_tetra10!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 10
	ed = 3
	for i=1:nq
		shape_tetra4shape_tetra10!(view(N, 1:nne, i), view(xi, 1:ed, i))
	end
	return (nne, nq)
end

function dshape_tetra10!(dNxi, xi::AbstractArray{<:AbstractFloat,1}=SIXTH*ones(3))
	"""
	dshape_tetra10!(dNxi, xi)
	"""
	(r,s,t) = xi

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

	return nl = (10, 3)
end

function dshape_tetra10!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_tetra10!!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 10
	ed = 3
	for i=1:nq
		dshape_tetra10!(view(dNxi, 1:nne, 1:ed, i), view(xi, 1:ed, i))
	end
	return nl = (nne, ed, nq)
end

#
# Eight node hexhedral element
#
function shape_hexa8!(N, xi::AbstractArray{<:AbstractFloat,1}=zeros(3))
	"""
	shape_hexa8!(N, xi)
	"""
  n=length(xi)
  I1=zeros(REALTYPE, n)
  I2=zeros(REALTYPE, n)
  for i=1:n
  	I1[i] = 0.5 - 0.5*xi[i]
  	I2[i] = 0.5 + 0.5*xi[i]
  end

  N[1] = I1[1]*I1[2]*I1[3]
  N[2] = I2[1]*I1[2]*I1[3]
  N[3] = I2[1]*I2[2]*I1[3]
  N[4] = I1[1]*I2[2]*I1[3]
  N[5] = I1[1]*I1[2]*I2[3]
  N[6] = I2[1]*I1[2]*I2[3]
  N[7] = I2[1]*I2[2]*I2[3]
  N[8] = I1[1]*I2[2]*I2[3]

  nl = 8

end # shape_hexa8

#
function shape_hexa8!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_hexa8!(N, xi)
	"""
	nq = size(xi,2)
	nne = 8
	ed = 3
	for i=1:nq
		shape_hexa8!(view(N, 1:nne, i), view(xi, 1:ed, i))
	end
	return (nne, nq)
end

function dshape_hexa8!(dNxi, coord::AbstractArray{<:AbstractFloat,1}=zeros(3))
	xi =   coord[1]
	eta =  coord[2]
	zeta = coord[3]

	dNxi[1,:] = EIGTH*[-1+eta+zeta-eta*zeta   -1+xi+zeta-xi*zeta  -1+xi+eta-xi*eta]
	dNxi[2,:] = EIGTH*[ 1-eta-zeta+eta*zeta   -1-xi+zeta+xi*zeta  -1-xi+eta+xi*eta]
	dNxi[3,:] = EIGTH*[ 1+eta-zeta-eta*zeta    1+xi-zeta-xi*zeta  -1-xi-eta-xi*eta]
	dNxi[4,:] = EIGTH*[-1-eta+zeta+eta*zeta    1-xi-zeta+xi*zeta  -1+xi-eta+xi*eta]
	dNxi[5,:] = EIGTH*[-1+eta-zeta+eta*zeta   -1+xi-zeta+xi*zeta   1-xi-eta+xi*eta]
	dNxi[6,:] = EIGTH*[ 1-eta+zeta-eta*zeta   -1-xi-zeta-xi*zeta   1+xi-eta-xi*eta]
	dNxi[7,:] = EIGTH*[ 1+eta+zeta+eta*zeta    1+xi+zeta+xi*zeta   1+xi+eta+xi*eta]
	dNxi[8,:] = EIGTH*[-1-eta-zeta-eta*zeta    1-xi+zeta-xi*zeta   1-xi+eta-xi*eta]

	nl = (8,3)

end # shape_hexa8

function dshape_hexa8!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_hexa8!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 8
	ed = 3
	for i=1:nq
		dshape_hexa8!(view(dNxi, 1:nne, 1:ed, i), view(xi, 1:ed, i))
	end
	return nl = (nne, ed, nq)
end

#
# Six node pentahedral element
#
# Tensor product of a CST with a [-1,1] Legendra basis in the zeta direction
function shape_penta6!(N, xi::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
	"""
	shape_penta6!(N, xi)
	"""
	(r,s,t) = xi
	N[1] = 0.5*(1 - r - s)*(1 + t)
	N[2] = 0.5*r*(1 - t)
	N[3] = 0.5*s*(1 - t)
	N[4] = 0.5*(1 - r - s)*(1 + t)
	N[5] = 0.5*r*(1 + t)
	N[6] = 0.5*s*(1 + t)
	nl = 6
end

function shape_penta6!(N, xi::AbstractArray{<:AbstractFloat,2})
	"""
	shape_penta6!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 6
	ed = 3
	for i=1:nq
		shape_penta6!(view(N, 1:nne, i), view(xi, 1:ed, i))
	end
	return (nne, nq)
end

function dshape_penta6!(dNxi, xi::AbstractArray{<:AbstractFloat,1}=THIRD*ones(2))
	"""
	dshape_penta6!(dNxi, xi)
	"""
	(r,s,t) = xi

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

function dshape_penta6!(dNxi, xi::AbstractArray{<:AbstractFloat,2})
	"""
	dshape_penta6!(dNxi, xi)
	"""
	nq = size(xi,2)
	nne = 6
	ed = 3
	for i=1:nq
		dshape_penta6!(view(dNxi, 1:nne, 1:ed, i), view(xi, 1:ed, i))
	end
	return nl = (nne, ed, nq)
end

#---------------------------------------------------------------------
#
#                  Q u a d r a t u r e   r u l e s
#
function qrule1d( npts )
	"""
	(qpt,qwt) = qrule1d(npts)
	"""
	if npts <= 1
		pts = [0.0]
		wts = [2.0]
	elseif npts == 2
		pts = 0.577350269189626*[-1; 1]
		wts = [1.0, 1.0]
	elseif npts == 3
		pts = 0.774596669241483*[-1; 0; 1]
		wts = [0.555555555555556, 0.888888888888889, 0.555555555555556]
	elseif npts == 4
		pts = [-0.861134311594053; -0.339981043584856; 0.339981043584856; 0.861134311594053]
		wts = [ 0.347854845137454,  0.652145154862546, 0.652145154862546, 0.347854845137454]
	elseif npts == 5
		pts = [-0.906179845938664; -0.538469310105683; 0.0; 0.538469310105683;
		        0.906179845938664]
		wts = [0.236926885056189, 0.478628670499366, 0.568888888888889,
		       0.478628670499366, 0.236926885056189]
	elseif npts == 6
		pts = [-0.932469514203152; -0.661209386466265; -0.238619186003152;
		        0.238619186003152; 0.661209386466265; 0.932469514203152]
		wts = [0.171324492379170, 0.360761573048139, 0.467913934572691,
		       0.467913934572691, 0.360761573048139, 0.171324492379170]
	elseif npts == 7
		pts = [-0.949107912342759; -0.741531185599394; -0.405845151377397; 0.0;
		       0.405845151377397; 0.741531185599394; 0.949107912342759]
		wts = [0.129484966168870, 0.279705391489277, 0.381830050505119,
		       0.417959183673469, 0.381830050505119, 0.279705391489277,
					 0.129484966168870]
	else # npts == 8
		if npts>8
			println("Quadrauture rule only defined to 8 points")
		end
		pts = [0-0.960289856497536; -0.796666477413627; -0.525532409916329;
		        -0.183434642495650; 0.183434642495650; 0.525532409916329;
						 0.796666477413627; 0.960289856497536]
		wts = [0.101228536290376, 0.222381034453374, 0.313706645877887,
		       0.362683783378362, 0.362683783378362, 0.313706645877887,
					 0.222381034453374, 0.101228536290376]
	end
	return (pts,wts)
end # function setqrule1d

function qrulecomp(sdim,npt1,npt2=npt1,npt3=npt1)
	"""
	(pts,wts) = qrulecomp(sdim,npt1,npt2=npt1,npt3=npt1)
	"""
	if sdim == 1
		npt2 = 1
		npt3 = 1
	elseif sdim == 2
		npt3 = 1
	end
	npts = npt1*npt2*npt3
	pts = zeros(sdim, npts)
	wts = zeros(npts)

	q = 1
	if sdim == 2
		pts1, wts1 = qrule1d(npt1)
		pts2, wts2 = qrule1d(npt2)
		for j = 1:npt2
			for i = 1:npt1
				pts[:, q] = [ pts1[i] pts2[j] ]
				wts[q] = wts1[i]*wts2[j]
				q += 1
			end
		end

	else # sdim == 3
		pts1, wts1 = qrule1d(npt1)
		pts2, wts2 = qrule1d(npt2)
		pts3, wts3 = qrule1d(npt3)
		for k = 1:npt3
			for j = 1:npt2
				for i = 1:npt1
					pts[:, q] = [ pts1[i] pts2[j]  pts3[k] ]
					wts[q] = wts1[i]*wts2[j]*wts3[k]
					q += 1
				end
			end
		end

	end  # of sdim if
	return (pts,wts)
end # function  qrulecomp

function qruletria(quadorder)
	"""
	(qpt,qwt) = qruletria(ord)
	"""

  if ( quadorder <= 1 )   # set quad points and quadweights
    quadpoint = [ 0.3333333333333; 0.3333333333333 ]
    quadweight = [1]

  elseif ( quadorder == 2 )
    quadpoint = [ 0.1666666666667  0.6666666666667  0.1666666666667;
				  0.1666666666667  0.1666666666667  0.6666666666667 ]
    quadweight = THIRD*[1, 1, 1]
  elseif ( quadorder <= 5 )
    quadpoint = [ 0.1012865073235 0.7974269853531 0.1012865073235 0.4701420641051 0.4701420641051 0.0597158717898 0.3333333333333;
	              0.1012865073235 0.1012865073235 0.7974269853531 0.0597158717898 0.4701420641051 0.4701420641051  0.3333333333333]

    quadweight = [0.1259391805448, 0.1259391805448,  0.1259391805448,
        0.1323941527885,  0.1323941527885, 0.1323941527885, 0.2250000000000]

  else
	quadpoint = [ 0.0651301029022 0.8697397941956 0.0651301029022 0.3128654960049
	      0.6384441885698 0.0486903154253 0.6384441885698 0.3128654960049
	      0.0486903154253 0.2603459660790 0.4793080678419 0.2603459660790 0.3333333333333;
                   0.0651301029022 0.0651301029022 0.8697397941956 0.0486903154253
	      0.3128654960049 0.6384441885698 0.0486903154253 0.6384441885698 0.3128654960049
		  0.2603459660790 0.2603459660790 0.4793080678419 0.3333333333333 ]
    quadweight = [0.0533472356088, 0.0533472356088,   0.0533472356088,
      0.0771137608903, 0.0771137608903, 0.0771137608903,  0.0771137608903,
      0.0771137608903,  0.0771137608903,  0.1756152576332,  0.1756152576332,
      0.1756152576332, -0.1495700444677 ]
  end
  quadweight=0.5*quadweight
	return (quadpoint,quadweight)
end

function qruletetra(ord)
	"""
	(qpt,qwt) = qruletetra(ord)
	"""
	if ord == 1
    quadpoint = [ 0.25 0.25 0.25 ]
    quadweight = [1.0]
	elseif ord == 2
    quadpoint = [ 0.58541020  0.13819660  0.13819660  0.13819660;
				  0.13819660  0.58541020  0.13819660  0.13819660;
				  0.13819660  0.13819660  0.58541020  0.13819660 ]
    quadweight = 0.25*[1, 1, 1, 1];
	else
    quadpoint = [ 0.25  0.50 SIXTH SIXTH  SIXTH;
				  0.25 SIXTH  0.50 SIXTH  SIXTH;
				  0.25 SIXTH SIXTH  0.50  SIXTH ]
    quadweight = 0.05*[-16, 9, 9, 9, 9]
	end
  return (quadpoint,quadweight)
end

function qrulepenta(ord)
	"""
	(qpt,qwt) = qrulepenta(ord)
	"""
	# get simplex rule
	(qptt, qwtt) = qruletria(ord)

	# and gauss rule
	n = Int32(round((ord+1)/2))
	(qptg, qwtg) = qrule1d(n)

	nq = length(qwtt)*length(qwtg)
	quadpoint = Array{REALTYPE,2}(undef, 3, nq)
	quadweight = Array{REALTYPE,1}(undef, nq)
	q = 0
	for g = 1:length(qwtg)
		zeta = qptg[g]
		for t = 1:length(qwtt)
			q += 1
			quadweight[q] = qwtt[t]*qwtg[g]
			quadpoint[1:2, q] = qptt[:, t]
			quadpoint[3, q] = zeta
		end
	end
  	return (quadpoint,quadweight)
end

function quadrule( qtype::String, sdim::Integer, p::Number )
	"""
	(qpt,qwt) = quadrule( qtype, sdim, p ) qtype="GAUSS", "SIMPLEX", or "PENTA"
	"""
	if qtype == "GAUSS"
		np = Int(ceil((p+1)/2))
		if sdim == 1
			return qrule1d(np)
		else
			return qrulecomp(sdim, np)
		end
	elseif qtype == "SIMPLEX"
		pint = Int(round(p))
		if sdim == 2
			return qruletria(pint)
		else
			return qruletetra(pint)
		end
	else # PENTA
		return qrulepenta(pint)
	end
end

#-----------------------------------------------------------------------------
#
#   Abstract Material
#
abstract type AbstractMaterial end
getprop(mat::AbstractMaterial, pname) = mat.prop[pname]

getindex(m::AbstractMaterial, p::String) = getindex(m.prop, p)
setindex!(m::AbstractMaterial, v::Number, p::String) = setindex!(m.prop, v, p)

struct GenMat <: AbstractMaterial
	prop::Dict{String,Any}
end
GenMat() = GenMat(Dict{String,REALTYPE}())

#-----------------------------------------------------------------------------
#
#   Abstract Property
#
abstract type AbstractProperty end
getprop(prop::AbstractProperty, pname) = prop.prop[pname]
getmatprop(prop::AbstractProperty, pname) = getprop(prop.mat, pname)

getindex(m::AbstractProperty, p::String) = getindex(m.prop, p)
setindex!(m::AbstractProperty, v::Number, p::String) = setindex!(m.prop, v, p)

struct GenProp <: AbstractProperty
	mat::AbstractMaterial
	prop::Dict{String,Any}
end
GenProp() = GenProp(GenMat(), Dict{String,REALTYPE}())
GenProp(prp::Dict{String,<:Any}) = GenProp(GenMat(), prp)
GenProp(mat::AbstractMaterial) = GenProp(mat, Dict{String,REALTYPE}())

#-----------------------------------------------------------------------------
#
#   Quadrature Data
#

struct QuadratureData
	qpt::Array{<:Number,2}
	qwt::Array{<:Number,1}
end
QuadratureData(qr::Tuple{Array{<:Number,2},Array{<:Number,1}}) = QuadratureData(qr[1], qr[2])
getqptwt(q::QuadratureData, i::Integer) = (q.qpt[:,i], q.qwt[i])
getindex(q::QuadratureData, i::Integer) = getqptwt(q,i)
length(qd::QuadratureData) = length(qd.qpt)
#-----------------------------------------------------------------------------
#
#   Abstract Element
#

# The AbstractElement seeks to abstract diffenent element implementations
# It contains an element connectivity
#
etopo2qtype = Dict("Line"=>"GAUSS", "Tria"=>"SIMPLEX", "Quad"=>"GAUSS",
     "Tetra"=>"SIMPLEX", "Hexa"=>"GAUSS", "Penta"=>"PENTA")

abstract type AbstractElement end
numelem(e::AbstractElement) = size(e.conn,2)
length(e::AbstractElement) = numelem(e)
size(e::AbstractElement) = size(e.conn)
numelem(e::Array{<:Int,2}) = size(e,2)
elemnne(e::AbstractElement) = size(e.conn,1)
elemnne(e::Array{<:Int,2}) = size(e,1)
elemsdim(e::AbstractElement) = elemedim(e)
elemvdim(e::AbstractElement) = elemsdim(e)
elemndofpn(e::AbstractElement) = 1
elemqrule(e::AbstractElement, p=elemorder(e)) = quadrule(etopo2qtype[elemtopo(e)], elemsdim(e), p )
getindex(e::AbstractElement, eid) =  getindex(e.conn, :, eid)
function setindex!(e::AbstractElement, econn::Array{<:Integer,1}, eid::Integer)
	n = min(length(econn), size(e.conn,1))
	return setindex!(e.conn, econn[1:n], 1:n, eid)
end

abstract type LineElement <: AbstractElement end
abstract type TriaElement <: AbstractElement end
abstract type QuadElement <: AbstractElement end
abstract type TetraElement <: AbstractElement end
abstract type HexaElement <: AbstractElement end
abstract type PentaElement <: AbstractElement end
elemtopo(e::LineElement) = "Line"
elemedim(e::LineElement) = 1
elemtopo(e::TriaElement) = "Tria"
elemedim(e::TriaElement) = 2
elemtopo(e::QuadElement) = "Quad"
elemedim(e::QuadElement) = 2
elemtopo(e::TetraElement) = "Tetra"
elemedim(e::TetraElement) = 3
elemtopo(e::HexaElement) = "Hexa"
elemedim(e::HexaElement) = 3
elemtopo(e::PentaElement) = "Penta"
elemedim(e::PentaElement) = 3

# New concrete element types can be constructed from these types
# Line2Element, Line3Element, Tria3Element ......
#
# Most functionality should be keep.  Only initialization need to be defined
# and elemsdim for elements where sdim != edim
abstract type Line2Element <: LineElement end
elemnne(e::Line2Element) = 2
elemorder(e::Line2Element) = 1
elemshape!(e::Line2Element, N, xi) = shape_line2!(N, xi)
elemdshape!(e::Line2Element, dN, xi) = dshape_line2!(dN, xi)

abstract type Line3Element <: LineElement end
elemnne(e::Line3Element) = 3
elemorder(e::Line3Element) = 2
elemshape!(e::Line3Element, N, xi) = shape_line3!(N, xi)
elemdshape!(e::Line3Element, dN, xi) = dshape_line3!(dN, xi)

abstract type Tria3Element <: TriaElement end
elemnne(e::Tria3Element) = 3
elemorder(e::Tria3Element) = 1
elemshape!(e::Tria3Element, N, xi) = shape_tria3!(N, xi)
elemdshape!(e::Tria3Element, dN, xi) = dshape_tria3!(dN, xi)

abstract type Tria6Element <: TriaElement end
elemnne(e::Tria6Element) = 6
elemorder(e::Tria6Element) = 2
elemshape!(e::Tria6Element, N, xi) = shape_tria6!(N, xi)
elemdshape!(e::Tria6Element, dN, xi) = dshape_tria6!(dN, xi)

abstract type Quad4Element <: QuadElement end
elemnne(e::Quad4Element) = 4
elemorder(e::Quad4Element) = 2
elemshape!(e::Quad4Element, N, xi) = shape_quad4!(N, xi)
elemdshape!(e::Quad4Element, dN, xi) = dshape_quad4!(dN, xi)

abstract type Tetra4Element <: TetraElement end
elemnne(e::Tetra4Element) = 4
elemorder(e::Tetra4Element) = 1
elemshape!(e::Tetra4Element, N, xi) = shape_tetra4!(N, xi)
elemdshape!(e::Tetra4Element, dN, xi) = dshape_tetra4!(dN, xi)

abstract type Tetra10Element <: TetraElement end
elemnne(e::Tetra10Element) = 10
elemorder(e::Tetra10Element) = 1
elemshape!(e::Tetra10Element, N, xi) = shape_tetra10!(N, xi)
elemdshape!(e::Tetra10Element, dN, xi) = dshape_tetra10!(dN, xi)

abstract type Hexa8Element <: HexaElement end
elemnne(e::Hexa8Element) = 8
elemorder(e::Hexa8Element) = 2
elemshape!(e::Hexa8Element, N, xi) = shape_hexa8!(N, xi)
elemdshape!(e::Hexa8Element, dN, xi) = dshape_hexa8!(dN, xi)

abstract type Penta6Element <: PentaElement end
elemnne(e::Penta6Element) = 6
elemorder(e::Penta6Element) = 1
elemshape!(e::Penta6Element, N, xi) = shape_penta6!(N, xi)
elemdshape!(e::Penta6Element, dN, xi) = dshape_penta6!(dN, xi)

# here is the generative code to do all concrete elements
# elements are given in FN
# They have the following construction methods
#
#    Element#D#()
#    Element#D#(conn, prop, data)
#    Element#D#(conn, prop)
#    Element#D#(numelem, prop, data)
#    Element#D#(numelem, prop)
#    Element#D#(numelem)
#
#  So for example  
#		elements = FEM.Hexa8D3(conn, prop)
#		elements = FEM.Line2D2(conn, prop) ...
#
FN = (:Line2D1, :Line2D2, :Line2D3,
            :Line3D1, :Line3D2, :Line3D3,
            :Tria3D2, :Tria3D3,
	    :Tria6D2, :Tria6D3,
	    :Quad4D2, :Quad4D3,
            :Tetra4D3, :Tetra10D3,
	    :Hexa8D3, :Penta6D3)
FP = (:Line2Element, :Line2Element, :Line2Element,
    :Line3Element, :Line3Element, :Line3Element,
    :Tria3Element, :Tria3Element,
    :Tria6Element, :Tria6Element,
    :Quad4Element, :Quad4Element,
    :Tetra4Element, :Tetra10Element,
    :Hexa8Element, :Penta6Element)
FNNE =  (2, 2, 2,   3, 3, 3,    3, 3,   6, 6,   4, 4,   4, 10, 8, 6)
FSDIM = (1, 2, 3,   1, 2, 3,    2, 3,   2, 3,   2, 3,   3,  3, 3, 3)
# parent nne sdim
for i = 1:length(FN)
    @eval begin
		struct $(FN[i]) <: $(FP[i])
			conn::Array{<:Integer,2}
			prop::AbstractProperty
			data::AbstractArray
            $(FN[i])(c, p, d) = size(c,1) == $(FNNE[i]) ? new(c, p, d) : error("Incorrect connectivity size")
        end
        $(FN[i])()  = $(FN[i])(Array{IDTYPE,2}(undef,$(FNNE[i]),0), GenProp(), [0.0])
        $(FN[i])(c::Array{<:Integer,2})  = $(FN[i])(c, GenProp(), [0.0])
        $(FN[i])(c::Array{<:Integer,2}, p::AbstractProperty)  = $(FN[i])(c, p, [0.0])
        $(FN[i])(ne::Integer, p=GenProp(), d=[0.0]) = $(FN[i])(Array{IDTYPE,2}(undef,$(FNNE[i]),ne), p, d)
        elemsdim(e::$(FN[i])) = $(FSDIM[i])
    end
end

#-----------------------------------------
struct ElementData
	nne::Integer
	edim::Integer
	sdim::Integer
	p::Integer
	qtype::String
	dofpn::Integer
	qrule::QuadratureData
	nqpt::Integer
	#vdim::Integer
end
function ElementData(e::AbstractElement, qord=elemorder(e))
	qr = QuadratureData(elemqrule(e,qord))
	return ElementData(elemnne(e), elemedim(e),
        elemsdim(e), elemorder(e), etopo2qtype[elemtopo(e)], elemndofpn(e),
		qr, length(qr.qwt))
end
local_dofs(e::ElementData) = collect(1:e.dofpn)

#---------------------------------------------------------------------
# Basic finite element utilities

function BTBop!(ke, B, a, add=true)
    """
    function BTBop!(ke, B, a, add=true)
        Computes ke += B^T*B*a 
		If add=false else ke is initially zeroed
    """
    m, n = size(B)
    if !add
		fill!(ke, 0.0)
	end
    for j=1:n
        for i=1:n
            for k=1:m
                ke[i,j] += B[k,i]*B[k,j]*a
            end
        end
    end
end

function BBTop!(ke, B, a, add=true)
    """
    function BBTop!(ke, B, a, add=true)
        Computes ke += B*B^T*a 
		If add=false else ke is initially zeroed
    """
    m, n = size(B)
    if !add
		fill!(ke, 0.0)
	end
    for k=1:n
        for j=1:m
            for i=1:m
                ke[i,j] += B[i,k]*B[j,k]*a
            end
        end
    end
end 

function BTCBop!(ke, B, C, a, add=true)
    """
    function BTCBop!(ke, B, C, a, add=true)
        Computes ke += B^T*C*B*a 
		If add=false else ke is initially zeroed
    """
    m, n = size(B)
    if !add
		fill!(ke, 0.0)
	end
    for l=1:m
        for k=1:m
            for j=1:n
                for i=1:n
                    ke[i,j] += B[k,i]*C[k,l]*B[l,j]*a
                end
            end
        end
    end
end

function BCBTop!(ke, B, C, a, add=true)
    """
    function BCBTop!(ke, B, C, a, add=true)
        Computes ke += B*C*B^T*a 
		If add=false else ke is initially zeroed
    """
    m, n = size(B)
    if !add
		fill!(ke, 0.0)
	end
    for k=1:n
        for l=1:n
            for j=1:m
                for i=1:m
                    ke[i,j] += B[i,k]*C[k,l]*B[j,l]*a
                end
            end
        end
    end
end

function setsctr!(sctr, conn, nn, ndofpn)
  """
  function setsctr(sctr, conn, nn, ndofpn)

    sets the array sctr to be a finite element
    type scatter vector.  The element connectivity
    is given in conn, and the number of dof per
    node is given as ndofpn.
  """
  i = 1
  for n=conn[1:nn]
    Ii = ndofpn*(n-1)+1
    for s=1:ndofpn
      sctr[i] = Ii
      i += 1
      Ii += 1
    end
  end
  ns = nn*ndofpn
end

function setsctr!(sctr, conn, nn, ldof, nldof, ndofpn)
    """
    function setsctr(sctr, conn, nn, ldof, nldof, ndofpn)

        sets the array sctr to be a finite element
        type scatter vector.  The element connectivity
        is given in conn, and the number of dof per
        node is given as ndofpn.
    """
    i = 1
    for n = conn[1:nn]
        Ii = ndofpn*(n-1)
        for s = ldof[1:nldof] 
            sctr[i] = Ii + s
            i += 1
        end
    end
	return i
end
function isojac!(jmat, dNdxi, coord, nn, sdim=3, edim=sdim)
  """
  function isojac(jmat, dNdxi, coord, nn, sdim=3, edim=sdim)
    Computes the Jacobian matrix for an isoparameteric mapping.
  """
  if (edim==3)
    jmat[1:sdim,1:edim] = coord[1:sdim,1:nn] * dNdxi[1:nn,1:edim]
    return

  elseif (edim==2)
    jmat[1:sdim,1:edim] = coord[1:sdim,1:nn] * dNdxi[1:nn,1:edim]
    if ( sdim == 3 )
      	jmat[1:sdim,3] = cross( jmat[1:sdim,1], jmat[1:sdim,2] )
      	jmat[1:sdim,3] = jmat[1:sdim,3]/norm(jmat[1:sdim,3])
    end
    return

  else #  edim = 1
    if ( sdim == 3 )
      	jmat[1:sdim,1:edim] = coord[1:sdim,1:nn] * dNdxi[1:nn,1:edim]
      	jmat[1:sdim,2] = cross( [0., 0., 1.], jmat[1:sdim,1] )
      	if ( norm(jmat[1:sdim,2])==0 )
        	jmat[1:sdim,2] = cross( [0., 1., 0.,], jmat[1:sdim,1] )
      	end
      	jmat[1:sdim,2] = jmat[1:sdim,2]/norm(jmat[1:sdim,2])
      	jmat[1:sdim,3] = cross( jmat[1:sdim,1], jmat[1:sdim,2] )
      	jmat[1:sdim,3] = jmat[1:sdim,3]/norm(jmat[1:sdim,3])

    elseif ( sdim == 2 )
      	jmat[1:sdim,1:edim] = coord[1:sdim,1:nn] * dNdxi[1:nn,1:edim]
      	jmat[1,2] = -jmat[2,1]
      	jmat[2,2] = jmat[1,1]
      	jmat[1:sdim,2] = jmat[1:sdim,2]/norm(jmat[1:sdim,2])

    else # sdim = 1
      	jmat[1:sdim,1:edim] = coord[1:sdim,1:nn] * dNdxi[1:nn,1:edim]
    end
  end
end # of function
isojac!(jmat, dNdxi, coord, edata::ElementData) = isojac!(jmat, dNdxi, coord, edata.nne, edata.sdim, edata.edim)

function gradbasis!(dNdx, dNdxi, coord, nn, sdim=3, edim=sdim)
  """
  function gradbasis(dNdxi, coord, nn, sdim=3, edim=sdim)

    computes dNdx and returns the determinant of the jacobian
    using an isoparametric mapping
  """
  s = zeros(3,3)
  isojac!(s, dNdxi, coord, nn, sdim, edim) # jac mat in dNdx
  if (sdim==1)
    dNdx[1:nn,1] = dNdxi[1:nn,1]/s[1,1]
    detj = s[1,1]
  elseif (sdim==2)
    detj = s[2,2]*s[1,1]-s[1,2]*s[2,1]
    idetj = 1.0/detj
    s[1:2,1:2] = [s[2,2] -s[1,2]; -s[2,1] s[1,1]]*idetj  # inv J
    dNdx[1:nn,1:sdim] = dNdxi[1:nn,1:edim]*s[1:edim,1:sdim]
  else # sdim == 3
    detj = -s[1,3]*s[2,2]*s[3,1] + s[1,2]*s[2,3]*s[3,1] + s[1,3]*s[2,1]*s[3,2] -
         s[1,1]*s[2,3]*s[3,2] - s[1,2]*s[2,1]*s[3,3] + s[1,1]*s[2,2]*s[3,3]
    idetj = 1.0/detj
    s[1:3,1:3] = [s[2,2]*s[3,3]-s[2,3]*s[3,2] s[1,3]*s[3,2]-s[1,2]*s[3,3] s[1,2]*s[2,3]-s[1,3]*s[2,2];
       s[2,3]*s[3,1]-s[2,1]*s[3,3] s[1,1]*s[3,3]-s[1,3]*s[3,1] s[1,3]*s[2,1]-s[1,1]*s[2,3];
       s[2,1]*s[3,2]-s[2,2]*s[3,1] s[1,2]*s[3,1]-s[1,1]*s[3,2] s[1,1]*s[2,2]-s[1,2]*s[2,1] ]*idetj  # inv J
    dNdx[1:nn,1:sdim] = dNdxi[1:nn,1:edim]*s[1:edim,1:sdim]
  end
  return detj
end # of function gradbasis
gradbasis!(dNdx, dNdxi, coord, edata::ElementData) = gradbasis!(dNdx, dNdxi, edata.nne, edata.sdim, edata.edim)

voigtdim = (1,3,6)
function formbmat!(B, dNdx, nn=size(dNdx,1), sdim=size(dNdx,2), vdim=size(B,1))
"""
function formbmat(B, dNdx, nn, sdim=3)
fills out the bmatrix.  The return duple is  the dimension of B
"""
	nc = nn*sdim

	if vdim == 1
		for i = 1:sdim
			B[1, i:sdim:nc] = dNdx[:, i]
		end
		return nc
	end

	for i = 1:sdim
	 	B[i,1:nc] = zeros(1,nc)
		B[i,i:sdim:nc] = dNdx[:,i]
	end

  	if sdim == 2
    	B[3,2:sdim:nc] = dNdx[:,1]
    	B[3,1:sdim:nc] = dNdx[:,2]
  	elseif sdim==3
    	B[4,1:sdim:nc] = zeros(1, nn)
    	B[4,2:sdim:nc] = dNdx[:,3]
    	B[4,3:sdim:nc] = dNdx[:,2]
    	B[5,1:sdim:nc] = dNdx[:,3]
    	B[5,2:sdim:nc] = zeros(1, nn)
    	B[5,3:sdim:nc] = dNdx[:,1]
    	B[6,1:sdim:nc] = dNdx[:,2]
    	B[6,2:sdim:nc] = dNdx[:,1]
    	B[6,3:sdim:nc] = zeros(1, nn)
  	end
  	return nc
end # function formbmat
formbmat!(B, dNdx, edata::ElementData) = formbmat!(B, dNdx, edata.nne, edata.sdim, voigtdim[edata.sdim])

function formnv!(Nv, N, sdim=size(Nv,2), nn=length(N))
	"""
	formnv!(Nv, N, sdim=size(Nv,2), nn=length(N))
	"""
	Ii = 0
	for I = 1:nn
		for i = 1:sdim
			Ii += 1
			Nv[Ii,i] = N[I]
		end
	end
	return Ii
end
formnv!(Nv, N, e::AbstractElement) = formnv!(Nv, N, elemsdim(e), elemnne(e))

function formnvt!(NvT, N, sdim=size(NvT,1), nn=length(N))
	"""
	formnvt!(NvT, N, sdim=size(NvT,1), nn=length(N))
	"""
	Ii = 0
	for I = 1:nn
		for i = 1:sdim
			Ii += 1
			NvT[i,Ii] = N[I]
		end
	end
	return Ii
end
formnvt!(NvT, N, e::AbstractElement) = formnvt!(NvT, N, elemsdim(e), elemnne(e))

# ----------- D E L A Y E D   A S S E M B L Y    S T U F F ------------
function delayassmij(conn, nldof=1)
    ne = size(conn,2)
    nne = size(conn,1)
    kedim = nldof*nne
    i = Array{IDTYPE}(undef,kedim,kedim,ne)
    j = Array{IDTYPE}(undef,kedim,kedim,ne)
    sctr = zeros(1,kedim)
    for e = 1:ne
        setsctr!(sctr, conn[:,e], nne, nldof)
        for (ii, Ii) in enumerate(sctr)
            for (jj, Jj) in enumerate(sctr)
                i[ii,jj,e] = Ii
                j[ii,jj,e] = Jj 
            end
        end
    end
    return i, j
end

function delayassm(kvals, conn, nldof)
    is, js = delayassmij(conn, nldof)
    return(sparse(vec(is), vec(js), vec(kvals)))
end

struct DelayedAssmMat
    is::Array{IDTYPE,3}
    js::Array{IDTYPE,3}
    kvals::Array{REALTYPE,3}
end

DelayedAssmMat(ne::Int, kdim::Int) = DelayedAssmMat(Array{IDTYPE}(undef,kdim,kdim,ne), 
                 Array{IDTYPE}(undef,kdim,kdim,ne), Array{REALTYPE}(undef,kdim,kdim,ne)) 

function DelayedAssmMat(conn, nldof) 
    is, js = delayassmij(conn, nldof)
    nne, ne = size(conn)
    d = nne*nldof
    DelayedAssmMat(is, js, zeros(d,d,ne))
end

function add_kmat!(K::DelayedAssmMat, e, ke)
    K.kvals[:,:,e] = ke
end

function add_kmat!(K::DelayedAssmMat, e, ke, sctr)
    K.kvals[:,:,e] = ke   
    for (ii, Ii) in enumerate(sctr)
        for (jj, Jj) in enumerate(sctr)
            K.is[ii,jj,e] = Ii
            K.js[ii,jj,e] = Jj 
        end
    end
end

function assemble_mat(K::DelayedAssmMat)
    return(sparse(vec(K.is), vec(K.js), vec(K.kvals)))
end

#----------------------------------------------------------------------------
# system operators and solving routines

function enforcebc!(K, f, ifix, ival)
	"""
	function enforcebc!(K, f, ifix, ival)

	Enforces an essential BC on a Kd=f system using an equation substitution
	"""
	m = size(K, 2)
	nfix = length(ifix)
	for ii = 1:nfix
		i = ifix[ii]
		di = ival[ii]
		Kii = K[i,i]
		for j = 1:m
			f[j] -= K[j,i]*di
			K[i,j] = 0.0
			K[j,i] = 0.0
		end
		K[i,i] = Kii
	end
	for ii = 1:nfix
		i = ifix[ii]
		f[i] = K[i,i]*ival[ii]
	end
end

function penaltybc!(K, f, ifix, ival=zeros(length(ifix)), penal=10.0e5)
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

function cholesky_solve(K, f)
	"""
	cholesky_solve(K::AbstractArray, f::Array{<:Number,1})
	"""
	Kr = cholesky(K)
	d = Kr\f
	return d
end

function fesolve!(K::Array, f, ifix, ival=zeros(length(ifix)))
	"""
	fesolve(K, f, ifix, ival=zeros) for dense arrays
	"""
	Kr = K[ifix,:]
	ff = deepcopy(f)
	enforcebc!(K, ff, ifix, ival)
  	d = cholesky_solve(K, ff)
  	f[ifix] = Kr*d
	return d
end # of fesolve function

function fesolve!(K::AbstractSparseArray, f, ifix, ival=zeros(length(ifix)))
	"""
	fesolve(K, f, ifix, ival=zeros) for sparse arrays
	This uses a iterative CG solver.  Could d better probably
	"""
	Kr = K[ifix,:]
	penaltybc!(K, f, ifix, ival)
	linsol = LinearProblem(K, f)
	sol = solve(linsol, IterativeSolversJL_CG())
	d = sol.u
	f[ifix,:] = Kr*d
	return d
end # of fesolve function

# ------------------------- D O F    S T U F F ----------------------
abstract type AbstractDofMap end
struct FixedDofMap <: AbstractDofMap
	ndpn::Integer
	nn::Integer
end
FixedDofMap() = FixedDofMap(1, 0)
FixedDofMap(nd::Integer) = FixedDofMap(nd, 0)
numDof(dmap::FixedDofMap) = dmap.nn*dmap.ndpn
GDOF(dmap::FixedDofMap, n::Integer, s::Integer=1) = dmap.ndpn*(n-1)+s

struct DofMap <: AbstractDofMap
	dmap::Dict{Tuple{Integer,Integer}, Integer}
end
DofMap() = DofMap(Dict{Tuple{IDTYPE, IDTYPE}, IDTYPE}())
function reorderDof!(dmap::DofMap, s::Integer=0)
	"""
	reorderDof!(dmap::DofMap, s::Integer=0)
	"""
	dm = sort(collect(dmap.dmap))
	for (i, ii) in dm
		s += 1
		dmap.dmap[i] = s
	end
end
function addDof!(dmap::DofMap, n::Integer, s::Integer)
	"""
	addDof!(dmap::DofMap, n::Integer, s::Integer)
	"""
	# does this dof exist?
	# if not add it
	Ii = length(dmap.dmap) + 1
	dmap.dmap[(n,s)] = Ii
end
numDof(dmap::DofMap) = length(dmap.dmap)
GDOF(dmap::DofMap, n::Integer, s::Integer=1) = dmap.dmap[(n,s)]

function setsctr!(sctr::Array{<:Integer,1}, dmap::AbstractDofMap,
	 	n::Array{<:Integer,1}, s::Array{<:Integer,1})
	"""
	etsctr!(sctr::Array{Integer,1}, dmap::AbstractDofMap, n::Array{Integer,1}, s::Array{Integer,1})
	"""
	ii = 0
	for ni in n
		for sj in s
			ii += 1
			sctr[ii] = GDOF(dmap, ni, sj)
		end
	end
	return ii
end
# ------------- B O U N D A R Y   C O N D I T I O N   S T U F F --------------
abstract type AbstractBC end
struct GeneralBC <: AbstractBC
	gdof::Array{IDTYPE,1}
	ival::Array{REALTYPE,1}
end
GeneralBC() = GeneralBC(Array{IDTYPE,1}(undef,0),
                            Array{REALTYPE,1}(undef,0))

function addBC!(bc::AbstractBC, gdof::Array{<:Integer,1},
			ival::Array{<:AbstractFloat,1})
	"""
	addBC!(bc::AbstractBC, gdof::Array{<:Integer,1},
				ival::Array{<:AbstractFloat,1})
	"""
	n2 = length(gdof)
	n1 = length(bc.gdof)
	nn = n1 + n2

	resize!(bc.gdof, nn)
	resize!(bc.ival, nn)
	for i = 1:n2
		bc.gdof[i+n1] = gdof[i]
		bc.ival[i+n1] = ival[i]
	end
	return 	nn
end

function addBC!(bc::AbstractBC, nid::Array{<:Integer,1},
			dmap::AbstractDofMap, ldof::Array{<:Integer,1},
			lval::Array{<:AbstractFloat,1})
	"""
	addBC!(bc::AbstractBC, nid::Array{<:Integer,1},
				dmap::AbstractDofMap, ldof::Array{<:Integer,1},
				lval::Array{<:AbstractFloat,1})
	"""
	n2 = length(nid) * length(ldof)
	n1 = length(bc.gdof)
	nn = n1 + n2

	resize!(bc.gdof, nn)
	resize!(bc.ival, nn)
	i = 0
	for n in nid
		for s in ldof
			i += 1
			bc.gdof[i+n1] = GDOF(dmap, n, s)
			bc.ival[i+n1] = lval[s]
		end
	end
	return 	nn
end

function addBC!(bc::AbstractBC, element::AbstractElement,
	node::Array{<:Number,2}, dmap::AbstractDofMap,
	ldof::Array{<:Integer,1}, lval::Array{<:Number,1})
	"""
	addBC!(bc::AbstractBC, element::AbstractElement,
		node::Array{<:Number,2}, dmap::AbstractDofMap,
		ldof::Array{<:Integer,1}, lval::Array{<:Number,1})
	"""

	ne = numelem(element)
	nne = elemnne(element)
	edim = elemedim(element)
	sdim = elemsdim(element)
	eorder = elemorder(element)
	ndofpn = length(ldof)
	p = 2*eorder
	(qpt, qwt) = elemqrule(element, p)
	nqpt = length(qwt)

	dNxi = zeros(REALTYPE, nne, edim, nqpt)
	dNdx = zeros(REALTYPE, nne, sdim)
	N = zeros(REALTYPE, nne, nqpt)
	sctr = zeros(IDTYPE, nne)
	elemdshape!(element, dNxi, qpt)
	elemshape!(element, N, qpt)

	fmap = Dict{IDTYPE, REALTYPE}()
	for e = 1:ne
		for n in element.conn[:,e]
			for s in ldof
				Ii = GDOF(dmap, n, s)
				fmap[Ii] = 0.0
			end
		end
	end

	for e = 1:ne
		econn = element.conn[:,e]
		coord = node[:, econn]
		me = zeros(REALTYPE, nne, nne)
		for q = 1:nqpt
			detj = gradbasis!(dNdx, dNxi[:,:,q], coord, nne, sdim, edim)
			me += N[:,q]*transpose(N[:,q])*(qwt[q]*detj)
		end
		fe = me*ones(nne,1)
		for s in ldof
			p = lval[s]
			for i = 1:nne
				Ii = GDOF(dmap, econn[i], s)
				fmap[Ii] += fe[i]*p
			end
		end
	end

	# put fmap into bc
	n1 = length(bc.ival)
	n2 = length(fmap)
	resize!(bc.gdof, n1+n2)
	resize!(bc.ival, n1+n2)
	n = n1
	for (Ii, ival) in fmap
		n += 1
		bc.gdof[n] = Ii
		bc.ival[n] = ival
	end
	return n
end

#----------------------------- IO SUBMODULE --------------------------
#                           Basic IO Capabilities
module IO

export output_mesh, output_node_sdata, output_node_vdata,
  output_element_sdata, write_outputfile

import ..FEM: AbstractElement, numelem, elemtopo
import ..FEM: Quad4D2, Quad4D3, Hexa8D3
import WriteVTK

topovtktype = Dict("Line"=>WriteVTK.VTKCellTypes.VTK_LINE,
                    "Tria"=>WriteVTK.VTKCellTypes.VTK_TRIANGLE,
					"Quad"=>WriteVTK.VTKCellTypes.VTK_QUAD,
				 	"Tetra"=>WriteVTK.VTKCellTypes.VTK_TETRA,
					"Hexa"=>WriteVTK.VTKCellTypes.VTK_HEXAHEDRON,
					"Penta"=>WriteVTK.VTKCellTypes.VTK_PENTAGONAL_PRISM)

elemvtktype(e::AbstractElement) = topovtktype[elemtopo(e)]
#elemvtktype(element::Truss3D2P) = WriteVTK.VTKCellTypes.VTK_LINE
#elemvtktype(element::FQuad4P) = WriteVTK.VTKCellTypes.VTK_QUAD
#elemvtktype(element::Quad4D2) = WriteVTK.VTKCellTypes.VTK_QUAD
#elemvtktype(element::Quad4D3) = WriteVTK.VTKCellTypes.VTK_QUAD
#elemvtktype(element::Hexa8D3) = WriteVTK.VTKCellTypes.VTK_HEXAHEDRON

function output_mesh(filename, node, element::AbstractElement)
	"""
	Writes a finite element type data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	output_mesh(filename, node, element)
	
	filename = output file name
	node = node coordinate array
	element is an AbstractElement type
	"""
	ne = numelem(element)
    cell =  Array{WriteVTK.MeshCell,1}(undef,ne)
    for e=1:ne
      cell[e] = WriteVTK.MeshCell(elemvtktype(element), element.conn[:,e])
    end
    vtkfile = WriteVTK.vtk_grid(filename, node, cell)
	return vtkfile
end

function output_mesh(filename, node, conn::AbstractArray, etype)
	"""	
	Writes a finite element type data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	output_mesh(filename, node, conn, etype)
	
	filename = output file name
	node = node coordinate array
	conn = connectivity array``
	etype = {"Line", "Tria", "Quad", "Tetra", "Hexa", "Penta"}
	"""
	nne, ne = size(conn)
	vtktype = topovtktype[etype]
    cell =  Array{WriteVTK.MeshCell,1}(undef,ne)
    for e=1:ne
      cell[e] = WriteVTK.MeshCell(vtktype, conn[:,e])
    end
    vtkfile = WriteVTK.vtk_grid(filename, node, cell)
	return vtkfile
end

function output_node_sdata(vtkfile, d, dataname)
	"""
	writes scalar nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().
	"""
	WriteVTK.vtk_point_data(vtkfile, d, dataname)
end

function output_node_vdata(vtkfile, d, dataname, ddofs=[1,2,3], stride=length(ddofs))
	"""
	writes vector nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	vtkfile a vtk output object. Typically comes from output_mesh()
	d nx1 array of nodal data
	dataname string of the dta name
	ddofs vector of nodal dofs, default is [1,2,3]
	stride stride between next node data, default is length(ddofs)
	"""
	# put d vector data in sdim x numnode array form
	nn = Int(floor(length(d)/stride))
	ns = length(ddofs)
	dv = reshape(d, (stride,nn))
	WriteVTK.vtk_point_data(vtkfile, dv[ddofs,:], dataname)
end

function output_element_sdata(vtkfile, d, dataname)
	"""
	writes vector nodal data to a vtk file opbject.  Note this does not actually
	write the file yet.  This is done in write_outputfile().

	vtkfile a vtk output object. Typically comes from output_mesh()
	d is a nex1 array of scalar element data
	dataname string of the data name
	"""
	WriteVTK.vtk_cell_data(vtkfile, d, dataname)
end

function write_outputfile(vtkfile)
	"""
	Writes the data to a file from a vtk object.
	
	function write_outputfile(vtkfile)
	"""
	outfiles = WriteVTK.vtk_save(vtkfile)
end
end # IO submodule
#--------------------------END OF IO SUBMODULE ---------------------------

# -------------------------  TESTS SUBMODULE -----------------------------
module Tests

export test1, test2

using ..FEM
using SparseArrays

function test1()
	(qpt, qwt) = FEM.quadrule("GAUSS", 1, 2)
	N = zeros(Float64, 2, length(qwt))
	nl = FEM.shape_line2!(N, qpt)
	dN = zeros(Float64, 2, 1, length(qwt))
	dnl = FEM.dshape_line2!(dN, qpt)
	(qpt, qwt) = FEM.quadrule("GAUSS", 2, 2)
	N = zeros(4, 4)
	FEM.shape_quad4!(N, qpt)
	dN = zeros(4,2,4)
	FEM.dshape_quad4!(dN, qpt)
end # function ex1

end # module FEMTests
#--------------------------------- END TESTS -----------------------------

end  # end of module
#-------------------------------------------------------------------------
#                E N D     O F     F E M    M O D U L E
#-------------------------------------------------------------------------
