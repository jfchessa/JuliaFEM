using MeshGen, FEM

# setup a simple hex 
#mesh = MeshGen.HexahedronMesh(10.0, 5.0, 2.0, 11, 6, 3)
mesh = MeshGen.HexahedronMesh(10.0, 5.0, 2.0, 101, 51, 21)
nodes = MeshGen.get_nodes(mesh);
conn = MeshGen.get_connectivity(mesh);

println(size(conn,2)," elements")

# first we will calculate the laplacian operator in the most simple way.
# We wil not assemble these since this is really the same process for all
# aproches. We will just calculate the element stiffness matices.
function laplace(nodes, conn)
    kappa = 1.0
    nne = 8
    sdim = 3
    ke = zeros(nne,nne)
    dNxi = zeros(nne,sdim) 
    dN = zeros(nne,sdim) 
    jmat = zeros(sdim,sdim)
    FEM.dshape_hexa8!(dNxi, [0., 0., 0.])
    for econn in eachcol(conn)
        detj = FEM.gradbasis!(dN, dNxi, (@view nodes[:,econn]), nne)
        FEM.BBTop!(ke, dN, kappa*detj)
    end
end

#laplace1(nodes,conn)
@time laplace(nodes,conn)
@time laplace(nodes,conn)

# Now we will calculate the laplacian operator with the Elemetn types.
# We wil not assemble these since this is really the same process for all
# aproches. We will just calculate the element stiffness matices.
mat = FEM.GenMat(Dict("kappa"=>1.0))
prop = FEM.GenProp(mat)
elements = FEM.Hexa8D3(conn, prop)
function laplace(nodes, elem::FEM.AbstractElement)
    kappa = elem.prop.mat["kappa"]
    nne = FEM.elemnne(elem)
    sdim = FEM.elemsdim(elem)
    ke = zeros(nne,nne)
    dNxi = zeros(nne,sdim) 
    dN = zeros(nne,sdim) 
    jmat = zeros(sdim,sdim)
    FEM.dshape_hexa8!(dNxi, [0., 0., 0.])
    for econn in eachcol(elem.conn)
         detj = FEM.gradbasis!(dN, dNxi, (@view nodes[:,econn]), nne)
         FEM.BBTop!(ke, dN, kappa*detj)
    end
end

@time laplace(nodes, elements)
@time laplace(nodes, elements)

# Do the assembly now
function laplace(nodes, conn, Kmat::FEM.DelayedAssmMat)
    kappa = 1.0
    nne = 8
    sdim = 3
    ke = zeros(nne,nne)
    dNxi = zeros(nne,sdim) 
    dN = zeros(nne,sdim) 
    jmat = zeros(sdim,sdim)
    FEM.dshape_hexa8!(dNxi, [0., 0., 0.])
    e=0
    for econn in eachcol(conn)
        detj = FEM.gradbasis!(dN, dNxi, (@view nodes[:,econn]), nne)
        FEM.BBTop!(ke, dN, kappa*detj)
        e += 1
        FEM.add_kmat!(Kmat, e, ke, econn)
    end
    K = FEM.assemble_mat(Kmat)
end

K = FEM.DelayedAssmMat(size(conn,2), 8)
@time  laplace(nodes, conn, K)
@time  laplace(nodes, conn, K)