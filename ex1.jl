module FEA2D

using Main.FEM, LinearAlgebra 

struct ProblemDefinition
"""
A structure to define an input Problem
"""
    node        # node coordinate array
    conn        # element connectivity (quad4) 
    E           # Young's modulus
    nu          # Poission's ratio
    thk         # the plate thickness
    ifix        # vector of fixed global dofs 
    iforce      # vector of global dofs where an external force is applied
    fforce      # vector of the external forces at the dofs in iforce.
end

function prob1()
"""
Defines a rectangular domain of LxH with the left hand side fixed and
a downward point load on the lower right hand corner.
"""
    L = 10
    H = 6
    nelx = 160
    nely = 80

    young = 10.0e6
    nu = 0.3
    thk = .25

    nnx = nelx+1
    nny = nely+1
    nn = nnx*nny

    conn = FEM.Meshing.genconn2d([1,2,nnx+2,nnx+1], nelx, nely);
    node = FEM.Meshing.nodearray2d([0 L L 0; 0 0 H H], nnx, nny);
    
    ifix = [collect(1:2*nnx:2*nn); collect(2:2*nnx:2*nn)]
    iforce = [2*nnx]
    fforce = -1000.0

    return ProblemDefinition(node, conn, young, nu, thk, ifix, iforce, fforce)
end

function bmat_quad4(coord, xi)
"""
 function B=bmat_quad4(coord,xi)

 Computes the strain-displacement matrix (B matrix) for a four node
 quadrilateral element.
      coord: the nodal coordinates of the element (4x2 matrix)
 function B,jac = bmat_quad4(coord,xi)
      Computes the B matrix and the Jacobian
"""    
    eta = xi[2]
    xi = xi[1]
        
    dNxi=0.25*[ -(1-eta) -(1-xi); (1-eta) -(1+xi); (1+eta)  (1+xi);  -(1+eta)  (1-xi) ]
        
    J = coord*dNxi
    jac = det(J)
    dN = dNxi/inv(J) # dN = dNxi*inv(J)
    
    B = [ dN[1,1] 0 dN[2,1] 0 dN[3,1] 0 dN[4,1] 0;
          0 dN[1,2] 0 dN[2,2] 0 dN[3,2] 0 dN[4,2];
        dN[1,2] dN[1,1] dN[2,2] dN[2,1] dN[3,2] dN[3,1] dN[4,2] dN[4,1] ]

    return B, jac
end

const SQR3RD = 0.5773502691896257
function kmat_quad4(coord, C, thk)
"""
ke = kmat_quad4(coord, C, thk)
    Generates equations for a plane stress 4 node quadralateral element, ke 
    C = material stiffness matrix
    thk = element thickness
    coord = coordinates at the element ends
"""
    ke = zeros(8,8)
    qpt = [-SQR3RD SQR3RD SQR3RD -SQR3RD; -SQR3RD -SQR3RD SQR3RD SQR3RD]
    for q=1:4
        xi = qpt[:,q]
        B, jac = bmat_quad4(coord,xi)
        ke = ke + transpose(B)*C*B*jac*thk
    end
    return ke
end

function compute_matstiff(prob)
    """
    Computes the plane stress material stiffness matrix
    """
    c1=prob.E/(1-prob.nu^2)  # plane stress material stiffness matrix
    c2=prob.nu*c1
    c3=0.5*(1-prob.nu)*c1
    return [c1 c2 0;c2 c1 0; 0 0 c3]
end

function compute_operators(prob)
    """
    Calculates the finite element operators, K and F.  K is a 
    delayed assembly (DelayedAssmMat)
    """
    nn = size(prob.node,2)
    ne = size(prob.conn,2)
    ndof = nn*2

    Fext = zeros(ndof,1)
    Fext[prob.iforce] .= prob.fforce

    Kdelay = FEM.DelayedAssmMat(prob.conn,2)
    ke = zeros(8,8)
    Ce = compute_matstiff(prob)
    for e = 1:ne
        ke = kmat_quad4(prob.node[:,prob.conn[:,e]], Ce, prob.thk)
        FEM.add_kmat!(Kdelay, e, ke)
    end

    return FEM.assemble_mat(Kdelay), Fext
end

function compute_stresses(prob, d)
    """
    Computes the element stresses. Each column of the return value
    are the Sxx Syy and Sxy stress components for the element corresponding
    to the row
    """
    nne, ne = size(prob.conn)
    ldof = 2
    Ce = compute_matstiff(prob)
    sig = zeros(3,ne)
    sctr = zeros(Int64,nne*ldof)
    for e = 1:ne
        FEM.setsctr!(sctr, prob.conn[:,e], nne, ldof)
        B, jac = bmat_quad4(prob.node[:,prob.conn[:,e]],[0. 0.])
        epse = B*d[sctr]
        sig[:,e] = Ce*epse
    end
    return sig
end

function solve(prob)
    """
    Solves the Problem
    """
    K, F = compute_operators(prob)
    d =  FEM.fesolve!(K, F, prob.ifix)
    sig = compute_stresses(prob, d)
    return d, sig
end

end  # of FEA2D module
##########################################################################

using .FEA2D

prob = FEA2D.prob1();
sol = FEA2D.solve(prob);
