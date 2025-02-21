__precompile__()

module MeshGen

export linspace, NullBias, LinearBias, PowerBias, BellcurveBias
export nodearray1d, nodearray2d, nodearray3d, genconn2d, genconn3d
export QuadrilateralMesh, HexahedronMesh

using FemBasics: REALTYPE, IDTYPE

"""
linspace(a,b,n) returns a vector in the same manner as the
Matlab linspace function.
"""
linspace(a,b,n) = collect(range(a,stop=b,length=n))
#linspace(a,b,n) = collect(LinRange(a,b,n))

"""
Bias structrues are used to setup non uniform point spaceings
"""
abstract type AbstractBias end
struct NullBias <: AbstractBias 
end
biaspts(xi, b::NullBias) = xi
struct LinearBias  <: AbstractBias
    s::REALTYPE
end
function biaspts(xi, b::LinearBias)
    n = length(xi)
    rv = b.s^(1/(n-2))
    d = 1
    for i in 2:n
        xi[i] = xi[i-1]+d
        d = d/rv
    end
    return xi/xi[n]
end

struct PowerBias  <: AbstractBias
    s::REALTYPE
end
function biaspts(xi, b::PowerBias)
    return xi.^(b.s)
end
struct BellcurveBias  <: AbstractBias
    s::REALTYPE
end
function biaspts(xi, b::BellcurveBias)
    n = length(xi)
    bc = 0.5*( tanh.(b.s*(xi.-0.5)) .+ 1 )
    bc = bc .- bc[1]
    return bc/bc[n]
end

function nodearray1d(p1, p2, n::Int, bias=NullBias())
"""
function nodearray1d(p1, p2, n::Int, bias=NullBias())

    Returns a set of, n, nodes lying on the line from points p1 to p2.
    The node spaceing can be biased with the parameter b1.

"""
    N2 = biaspts(linspace(0, 1, n),bias)
    N1 = 1 .- N2
	s = length(p1)
	x = zeros(s,Int(n))
	for i = 1:s
		x[i,:] = N1*p1[i] + N2*p2[i]
	end
	return x
end 

function nodearray2d( corners, n1::Int, n2::Int, 
            b1=NullBias(), b2=NullBias() )
"""
function nodearray2d(corners, n1, n2, b1, b2)

    Returns a set of, n1 by n2, nodes lying on the quadrilateral
    formed by the points given by corners. The node spaceing can be 
    biased with the parameters b1 and b2.

"""
	# corners is the coordinates of the corners of the 2D patch
	# in column format (i.e. each node is a column)
	sd = size(corners,1)
    nn = Int(n1*n2)
	x = zeros(sd,nn)
	xi = biaspts(linspace(-1, 1, n1), b1)
	eta = biaspts(linspace(-1, 1, n2), b2)

	n=1
	N = zeros(4)
	for j=1:n2
	  for i=1:n1
	    s = xi[i]; t = eta[j]
	    N[1] = 0.25*(1-s)*(1-t)
	    N[2] = 0.25*(1+s)*(1-t)
        N[3] = 0.25*(1+s)*(1+t)
        N[4] = 0.25*(1-s)*(1+t)
	    x[1:sd, n] = corners*N
	    n += 1
	  end
	end
	return x
end 

function nodearray3d(corners, n1::Int, n2::Int, n3::Int, 
    b1=NullBias(), b2=NullBias(), b3=NullBias())
"""
function nodearray2d(corners, n1, n2, n3, b1, b2, b3)

    Returns a set of, n1xn2xn3, nodes lying on the hexahedron
    formed by the points given by corners. The node spaceing can be 
    biased with the parameters b1, b2 and b3.
    
"""
	# corners is the coordinates of the corners of the 3D hexahedral region
	# in column format (i.e. each node is a column)
	sd = size(corners,1)
  	nn = n1*n2*n3
	x = zeros(sd,nn)

	xi = biaspts(linspace(-1, 1, n1), b1)
	eta = biaspts(linspace(-1, 1, n2), b2)
	zeta = biaspts(linspace(-1, 1, n3), b3)

	n = 1
	N = zeros(8)
	for k=1:n3
        I13 = 0.5 - 0.5*zeta[k]
        I23 = 0.5 + 0.5*zeta[k]
	    for j=1:n2
            I12 = 0.5 - 0.5*eta[j]
            I22 = 0.5 + 0.5*eta[j]
	  	    for i=1:n1
                I11 = 0.5 - 0.5*xi[i]
                I21 = 0.5 + 0.5*xi[i]
          
                N[1] = I11*I12*I13
                N[2] = I21*I12*I13
                N[3] = I21*I22*I13
                N[4] = I11*I22*I13
                N[5] = I11*I12*I23
                N[6] = I21*I12*I23
                N[7] = I21*I22*I23
                N[8] = I11*I22*I23
          
	    	    x[1:sd, n] = corners*N
	    	    n += 1
		    end
	    end
	end
	return x
end # function

function genconn2d(node_pattern, num_u, num_v=num_u, inc_u=1, inc_v=2)
"""
function genconn2d(node_pattern, num_u, num_v=num_u, inc_u=1, inc_v=2)

    Returns a connectivity matrix on a 2d grid.

    node_pattern = the initial connectivity node_pattern
    num_u = the number of elements in the u-direction
    num_v = the number of elements in the v-direction
    inc_u = the increment in the element connectivity when going from
            one element to the next in the u-direction.  The default
            is inc_u=1
    inc_v = the increment of the element connectivity when going from
            the last element in the previous v-row to the next v-row.
            The default is inc_v = 2

"""
	ne = num_u*num_v
	nnp = length(node_pattern)

	element = zeros(IDTYPE,nnp,ne)
	inc = 0
	e = 1
	for row = 1:num_v
	   for col = 1:num_u
	      element[1:nnp,e] = node_pattern .+ inc
	      inc += inc_u
	      e += 1
	   end
	   inc += (inc_v - inc_u)
	end
	return element
end # genconn2d function

function genconn3d(node_pattern, num_u, num_v=num_u, num_w=num_v,
	     inc_u=1, inc_v=2, inc_w=num_u+3)
"""
function genconn3d(node_pattern, numu, numv, numw, incu, incv, incw)
    
    Returns a connectivity matrix on a (u, v, w) 3d grid.

    node_pattern = the initial connectivity node_pattern
    num_u = the number of elements in the u-direction
    num_v = the number of elements in the v-direction
    num_w = the number of elements in the w-direction
    inc_u = the increment in the element connectivity when going from
            one element to the next in the u-direction.  The default
            is inc_u=1
    inc_v = the increment of the element connectivity when going from
            the last element in the previous v-row to the next v-row.
            The default is inc_v = 2
    inc_w = the increment of the element connectivity when going from
            the last element in the previous w-level to the next w=level.
            The default is inc_w = num_u+3

"""
	ne = num_u*num_v*num_w
	nnp = length(node_pattern)

	element = zeros(IDTYPE,nnp,ne)
	inc = 0
	e = 1
	for k = 1:num_w
		for j = 1:num_v
	   	for i = 1:num_u
	      element[1:nnp,e] = node_pattern .+ inc
	      inc += inc_u
	      e += 1
	   	end
	   inc += (inc_v - inc_u)
	 	end
		inc += (inc_w-inc_v)
	end
	return element
end # genconn3d function

########################################################################
#
#  Some structures that hold some basic meshes
#
struct QuadrilateralMesh
    corners::Array{REALTYPE,2}
    ne::Array{Integer}
    bias::Tuple{<:AbstractBias, <:AbstractBias}
end

QuadrilateralMesh(l1, l2, n1, n2) =
	QuadrilateralMesh([0. l1 l1 0.;0. 0. l2 l2], [n1, n2],
                     (NullBias(),NullBias()))

QuadrilateralMesh(l1, l2, n1, n2, b1, b2) =
    QuadrilateralMesh([0. l1 l1 0.;0. 0. l2 l2], [n1, n2],
        (b1, b2))

function get_nodes(reg::QuadrilateralMesh)
    return nodearray2d(reg.corners, reg.ne[1]+1, reg.ne[2]+1, reg.bias[1],
             reg.bias[2])
end

function get_connectivity(reg::QuadrilateralMesh)
	n1 = reg.ne[1] + 1
	return genconn2d([1 2 n1+2 n1+1], reg.ne[1], reg.ne[2])
end

function get_vertex_node(reg::QuadrilateralMesh, vid)
	if vid==1
		return 1
	elseif vid==2
		return reg.ne[1]+1
	elseif vid==3
		return (reg.ne[1]+1)*(reg.ne[2]+1)
	else #vid ==4
		return (reg.ne[1]+1)*(reg.ne[2]+1) - reg.ne[1]
	end
end

function get_edge_connectivity(reg::QuadrilateralMesh, eid)
	if eid==1
		return genconn2d([1 2], reg.ne[1], 1)
	elseif eid==2
		na = get_vertex_node(reg::QuadrilateralMesh, 2)
		return genconn2d([na 2*na], reg.ne[2], 1, na)
	elseif eid==3
		na = get_vertex_node(reg::QuadrilateralMesh, 3)
		return genconn2d([na na-1], reg.ne[1], 1, -1)
	else # eid==4
		na = get_vertex_node(reg::QuadrilateralMesh, 2)
		nb = get_vertex_node(reg::QuadrilateralMesh, 4)
		return genconn2d([nb nb-na], reg.ne[2], 1, -na)
	end
end

#-----------------------------------------------------------------------
struct HexahedronMesh
    corners::Array{REALTYPE,2}
    ne::Array{Integer}
    bias::Tuple{<:AbstractBias, <:AbstractBias, <:AbstractBias}
end

HexahedronMesh(l1, l2, l3, n1, n2, n3) =
  HexahedronMesh([0. l1 l1 0. 0. l1 l1 0.;
                  0. 0. l2 l2 0. 0. l2 l2;
				  0. 0. 0. 0. l3 l3 l3 l3],[n1, n2, n3],
                  (NullBias(), NullBias(), NullBias()))

function get_nodes(reg::HexahedronMesh)
    return nodearray3d(reg.corners, reg.ne[1]+1, reg.ne[2]+1, reg.ne[3]+1)
end

function get_connectivity(reg::HexahedronMesh)
	n = (reg.ne[1]+1)*(reg.ne[2]+1)
	return genconn3d([1 2 reg.ne[1]+3  reg.ne[1]+2  1+n 2+n reg.ne[1]+3+n  reg.ne[1]+2+n],
	               reg.ne[1], reg.ne[2], reg.ne[3])
end

function get_vertex_node(reg::HexahedronMesh, vid)
	nn12 = (reg.ne[1]+1)*(reg.ne[2]+1)
	incz = nn12*(reg.ne[3])
	if vid==1
		return 1
	elseif vid==2
		return reg.ne[1]+1
	elseif vid==3
		return nn12
	elseif vid==4
		return nn12 - reg.ne[1]
	elseif vid==5
		return 1 + incz
	elseif vid==6
		return reg.ne[1]+1 + incz
	elseif vid==7
		return nn12 + incz
	else # vid == 8
		return nn12 - reg.ne[1] + incz
	end
end

function get_edge_connectivity(reg::HexahedronMesh, eid)
	nn1 = reg.ne[1]+1
	nn2 = reg.ne[2]+1
	nn3 = reg.ne[3]+1
	if eid==1
		inc = 1
		n = nn1-1
		na = get_vertex_node(reg::HexahedronMesh, 1)
		nb = na + 1
	elseif eid==2
		na = get_vertex_node(reg::HexahedronMesh, 2)
		nb = na + nn1
		inc = nn1
		n = nn2-1
	elseif eid==3
		na = get_vertex_node(reg::HexahedronMesh, 3)
		nb = na - 1
		inc = -1
		n = nn1-1
	elseif eid==4
		inc = -nn1
		n = nn2-1
		na = get_vertex_node(reg::HexahedronMesh, 4)
		nb = na + inc
	elseif eid==5
		inc = 1
		n = nn1-1
		na = get_vertex_node(reg::HexahedronMesh, 5)
		nb = na + inc
	elseif eid==6
		inc = nn1
		n = nn2-1
		na = get_vertex_node(reg::HexahedronMesh, 6)
		nb = na + inc
	elseif eid==7
		inc = -1
		n = nn1-1
		na = get_vertex_node(reg::HexahedronMesh, 7)
		nb = na + inc
	elseif eid==8
		inc = -nn1
		n = nn2-1
		na = get_vertex_node(reg::HexahedronMesh, 8)
		nb = na + inc
	elseif eid==9
		inc = nn1*nn2
		n = nn3-1
		na = get_vertex_node(reg::HexahedronMesh, 1)
		nb = na + inc
	elseif eid==10
		inc = nn1*nn2
		n = nn3-1
		na = get_vertex_node(reg::HexahedronMesh, 2)
		nb = na + inc
	elseif eid==9
		inc = nn1*nn2
		n = nn3-1
		na = get_vertex_node(reg::HexahedronMesh, 3)
		nb = na + inc
	elseif eid==9
		inc = nn1*nn2
		n = nn3-1
		na = get_vertex_node(reg::HexahedronMesh, 4)
		nb = na + inc
	end
	return genconn2d([na nb], n, 1, inc)
end

function get_face_connectivity(reg::HexahedronMesh, fid)
	n1 = reg.ne[1]+1
	n2 = reg.ne[2]+1
	n3 = reg.ne[3]+1
    n12 = n1*n2
    if fid==1
        ptrn = [n1 2*n1 2*n1+n12 n1+n12]
        inc1 = n1
        inc2 = 2*n1
        nn1 = n2
        nn2 = n3
    elseif fid==-1
	    ptrn = [1 n12+1 n12+n1+1 n1+1]
        inc1 = n1
        inc2 = 2*n1
        nn1 = n2
        nn2 = n3
    elseif fid==2
        c1 = n12-n1+1
        ptrn = [c1+n1 c1+n1+1 c1+1 c1]
        ptrn = [c1 c1+n12 c1+n12+1 c1+1]
        inc1 = 1
        inc2 = n12-n1+2
        nn1 = n1
        nn2 = n3
    elseif fid==-2
        ptrn = [1 2 n12+2 n12+1]
        inc1 = 1
        inc2 = n12-n1+2
        nn1 = n1
        nn2 = n3
    elseif fid==3
        c1 = n1*n2*(n3-1)+1
        ptrn = [c1 c1+1 c1+n1+1 c1+n1]
        inc1 = 1
        inc2 = 2
        nn1 = n1
        nn2 = n2
    elseif fid==-3
        ptrn = [n1+1 n1+2 2 1]
        inc1 = 1
        inc2 = 2
        nn1 = n1
        nn2 = n2
    end
    return genconn2d(ptrn, nn1-1, nn2-1, inc1, inc2)
end

function get_nodeids(conn)
    return collect(Set(conn))
end

function get_edge_nodeids(reg, eid)
	return get_nodeids(get_edge_connectivity(reg,eid))
end

function get_face_nodeids(reg, eid)
	return get_nodeids(get_face_connectivity(reg,eid))
end

function afine!(node, A, b)
	# node -> A*node + b
	if size(b,1) != size(node,1)
		v = b'
	else
		v = b
	end
	nn = size(node,2)
	for i=1:nn
		node[:,i] = A*node[:,i] + v
	end
end

end 
# --------------------- END OF MeshGen MODULE -------------------------#