
# ------------------------  MESHING SUBMODULE ---------------------------
module MeshGen

export nodearray1d, nodearray2d, nodearray3d, genconn2d, genconn3d
export QuadrilateralMesh, HexahedronMesh
export get_nodes, get_connectivity, get_vertex_node, get_edge_connectivity,
    get_face_connectivity, get_nodeids, get_edge_nodeids, get_face_nodeids
export afine!

REALTYPE = Float32
IDTYPE = Int32

#linspace(x1,x2,n) = LinRange(x1, x2, n)
# function linspace(x1, x2, n)
# 	return Array(range(x1, stop=x2, length=n))
# end

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

function nodearray1d(p1, p2, n)
	N2 = LinRange(0, 1, Int(n))
	N1 = 1 .- N2
	s = length(p1)
	x = zeros(s,Int(n))
	for i = 1:s
		x[i,:] = N1*p1[i] + N2*p2[i]
	end
	return x
end # function

function nodearray2d(corners, n1, n2=n1)
	# corners is the coordinates of the corners of the 2D patch
	# in column format (i.e. each node is a column)
	s = size(corners,1)
    nn = Int(n1*n2)
	x = zeros(s,nn)
	xi = LinRange(-1, 1, Int(n1))
	eta = LinRange(-1, 1, Int(n2))

	n=1
	N = zeros(4)
	for j=1:n2
	  for i=1:n1
	    pt = [xi[i], eta[j]]
	    shape_quad4!(N, pt)
	    x[1:s, n] = corners*N
	    n += 1
	  end
	end
	return x
end # function


function nodearray3d(corners, nn1, nn2=n1, nn3=n2)
	# corners is the coordinates of the corners of the 3D hexahedral region
	# in column format (i.e. each node is a column)
	s = size(corners,1)
	n1 = Int(floor(nn1))
	n2 = Int(floor(nn2))
	n3 = Int(floor(nn3))
  	nn = n1*n2*n3
	x = zeros(s,nn)

	xi = LinRange(-1, 1, n1)
	eta = LinRange(-1, 1, n2)
	zeta = LinRange(-1, 1, n3)

	n = 1
	N = zeros(8)
	for k=1:n3
	  for j=1:n2
	  	for i=1:n1
	    	pt = [xi[i], eta[j], zeta[k]]
	    	shape_hexa8!(N, pt)
	    	x[1:s, n] = corners*N
	    	n += 1
		end
	  end
	end
	return x
end # function

function genconn2d(node_pattern, num_u, num_v=num_u, inc_u=1, inc_v=2)
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

struct QuadrilateralMesh
    corners::Array{REALTYPE,2}
    ne::Array{Integer}
end

QuadrilateralMesh(l1, l2, n1, n2) =
	QuadrilateralMesh([0. l1 l1 0.;0. 0. l2 l2], [n1, n2])

function get_nodes(reg::QuadrilateralMesh)
    node = nodearray2d(reg.corners, reg.ne[1]+1, reg.ne[2]+1)
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

struct HexahedronMesh
    corners::Array{REALTYPE,2}
    ne::Array{Integer}
end

HexahedronMesh(l1, l2, l3, n1, n2, n3) =
  HexahedronMesh([0. l1 l1 0. 0. l1 l1 0.;
                  0. 0. l2 l2 0. 0. l2 l2;
				  0. 0. 0. 0. l3 l3 l3 l3],[n1, n2, n3])

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

end # of MeshGen module
# ---------------------  END OF MESHING SUBMODULE -----------------------
