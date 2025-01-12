__precompile__()

module StructuralElements

using FemBasics: REALTYPE, IDTYPE, BTCBop!,
                 GAUSS2D_2PT, GAUSS2D_2WT

using LinearAlgebra
using StaticArrays

export truss2d2_kmat!, truss2d2_bmat!, beam2d2_kmat!, beam2d2_bmat!
export truss2d2_mmat!, truss3d2_mmat!
export truss3d2_kmat!, truss3d2_bmat!, beam3d2_kmat!, beam3d2_bmat!
export mat1_cmat

# This is globally allocated space for doing local calculations
# I know this is not a great idea, but does really speed things up
# Use views to help address this memory in the functions.
const global _ScRaTcH_REAL_ = zeros(REALTYPE,96,96)

########################################################################
# Stiffness and B matrices for several basic linear structural 
# elements in 2d and 3d.
########################################################################
function mat1_cmat(E, nu, form="3D")
"""
    function mat1_cmat(E, nu, form="3D")

    Returns the material stiffness matrix for a linear isotropic elastic
    material (MAT1 in Nstran).

    E = Young's modulus
    nu = Poisson's ratio
    form = Stress formulation the options are as below.
        AXIAL - for axial stiffness
        SHEAR - for shear stiffness
        PSTRESS - Plane stress 
        PSTRAIN - Plane strain 
        AXISYM - Axisymmetric
        3D (default) - Full three-dimensional case
"""
    if form == "AXIAL"

        cmat =  ones(REALTYPE,1,1)*E

    elseif form == "SHEAR"

        cmat = ones(REALTYPE, 1, 1)*(0.5*E/(1+nu))

    elseif form == "PSTRESS"
        
        c1 = REALTYPE(E/(1-nu^2))   
        c2 = REALTYPE(nu*c1)
        c3 = REALTYPE(0.5*(1-nu)*c1)
        cmat = [c1 c2 0; c2 c1 0; 0 0 c3]
        
    elseif form == "PSTRAIN"
        
        c0 = REALTYPE(E/(1-2*nu)/(1+nu))
        c1 = REALTYPE((1-nu)*c0)   
        c2 = REALTYPE(nu*c0)   
        c3 = REALTYPE(0.5*(1-2*nu)*c0)

        cmat = [c1 c2 0; c2 c1 0; 0 0 c3]

    elseif form == "AXISYM"
        
        c0 = REALTYPE(E/(1-2*nu)/(1+nu))
        c1 = REALTYPE((1-nu)*c0)   
        c2 = REALTYPE(nu*c0)   
        c3 = REALTYPE(0.5*(1-2*nu)*c0)

        cmat = [c1 c2 c2 0; 
                c2 c1 c2 0; 
                c2 c2 c1 0; 
                 0  0  0 c3]

    else

        c0 = REALTYPE(E/(1-2*nu)/(1+nu))
        c1 = REALTYPE((1-nu)*c0)   
        c2 = REALTYPE(nu*c0)   
        c3 = REALTYPE(0.5*(1-2*nu)*c0)

        cmat = [c1 c2 c2  0  0  0; 
                c2 c1 c2  0  0  0; 
                c2 c2 c1  0  0  0; 
                 0  0  0 c3  0  0;
                 0  0  0  0 c3  0; 
                 0  0  0  0  0 c3]

    end

    return cmat
end

# Local use
function release_dof!(ke, pp::Array{<:Int}, off::Int=0)
    """
        function release_dof!(ke, pp, off=0)
    
            Releases the dofs of a stiffness matrix by zeroing out the 
            assoicated rows and columns
    
            ke = the matrix to be zeroed
            pp = a vector of integers associated with the dof to be released
                 0 < pp[i] <= 7
            off = an integer that offsets the local dof in (for nodes that 
                are not the first)
    """
    for i in pp
        if i > 0
            for ii in 1:6
                ke[ii,i+off] = 0.0
                ke[i+off,ii] = 0.0
            end
        end
    end
end

#----------------------------------------------------------------------
# TRUSS2D2 Element - two node truss element in 2D
function truss2d2_kmat!(ke, coord, AE)
"""
     truss2d2_kmat!(ke, coord, AE)  
     Generates stiffness matrix of a 2 node 2D truss element with 
     axial stiffness.
        ke = Returns the 4x4 element stiffness matrix 
     	AE = modulus of elasticity times the cross sectional area
     	coord = coordinates at the element ends in column format.  The 
                first column is the first node and second column holds 
                the second node.
"""    
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL = 1/norm(n2-n1)
    c = (n2[1]-n1[1])*invL
    s = (n2[2]-n1[2])*invL
    ka = (AE*invL)
    c2 = c*c*ka
    s2 = s*s*ka
    cs = c*s*ka
    ke[1,1] =  c2;  ke[1,2] =  cs;   ke[1,3] = -c2;   ke[1,4] = -cs; 
    ke[2,1] =  cs;  ke[2,2] =  s2;   ke[2,3] = -cs;   ke[2,4] = -s2; 
    ke[3,1] = -c2;  ke[3,2] = -cs;   ke[3,3] =  c2;   ke[3,4] =  cs; 
    ke[4,1] = -cs;  ke[4,2] = -s2;   ke[4,3] =  cs;   ke[4,4] =  s2; 
end

function truss2d2_bmat!( B, coord ) 
"""
    truss2d2_bmat( be, coord )  
    Generates the B-matrix of a 2 node 2D truss element with axial 
    stiffness.
        be = Returns the 1x4 element B-matrix 
  	    coord =  coordinates at the element ends in column format.  
                 The first column is the first node and second column 
                 holds the second node.
"""
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL2 = (1/norm(n2-n1))^2
    c = (n2[1]-n1[1])*invL2
    s = (n2[2]-n1[2])*invL2
    B[1,1] = -c; B[1,2] = -s; B[1,3] = c; B[1,4] = s;
end

function truss2d2_mmat!(me, coord, rhoA)
"""
    Function truss2d2_mmat!(me, coord, rhoA)

    Computes the consistent mass matrix for a 2 node truss element in
    2-dimensions.
        me = the 4x4 consistent mass matrix on return
  	    coord =  coordinates at the element ends in column format.  
                 The first column is the first node and second column 
                 holds the second node.
        rhoA =  the specific density, ie.e. the density times the 
                cross-sectional area
"""
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    rhoAL = rhoA*norm(n2-n1)
    mii = rhoAL/3
    mij = rhoAL/6        
    me[1,1] = mii;  me[1,2] = 0.0;  me[1,3] = mij;  me[1,4] = 0.0; 
    me[2,1] = 0.0;  me[2,2] = mii;  me[2,3] = 0.0;  me[2,4] = mij; 
    me[3,1] = mij;  me[3,2] = 0.0;  me[3,3] = mii;  me[3,4] = 0.0; 
    me[4,1] = 0.0;  me[4,2] = mij;  me[4,3] = 0.0;  me[4,4] = mii;  
end

#----------------------------------------------------------------------
# TRUSS3D2 Element - two node truss element in 3D
function truss3d2_kmat!( ke, coord, AE, JG )
"""
    truss3d2_kmat!( ke, coord, AE, JG )  
    Generates stiffness matrix of a 2 node 3D truss element with 
    torsional stiffness (similar to a NASTRAN CROD element)
        ke = Returns the 12x12 element stiffness matrix 
     	AE = modulus of elasticity
     	JG = Area of cross-section
  	    coord =  coordinates at the element ends in column format.  
                 The first column is the first node and second column 
                 holds the second node.
""" 
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL = (1/norm(n2-n1))
    nx = (n2[1]-n1[1])*invL
    ny = (n2[2]-n1[2])*invL  
    nz = (n2[3]-n1[3])*invL  
    
    ka = (AE*invL)
    kt = (JG*invL)

    n11 = nx*nx;  n12 = nx*ny;  n13 = nx*nz;
                  n22 = ny*ny;  n23 = ny*nz;
                                n33 = nz*nz;
    fill!(ke, 0.0)
    kv = @view ke[1:6,1:6]
    kv[1,1] = ka*n11;     kv[1,2] = ka*n12;     kv[1,3] = ka*n13;  
    kv[2,1] = kv[1,2];    kv[2,2] = ka*n22;     kv[2,3] = ka*n23; 
    kv[3,1] = kv[1,3];    kv[3,2] = kv[2,3];    kv[3,3] = ka*n33; 
    kv[4,4] = kt*n11;     kv[4,5] = kt*n12;     kv[4,6] = kt*n13; 
    kv[5,4] = kv[4,5];    kv[5,5] = kt*n22;     kv[5,6] = kt*n23; 
    kv[6,4] = kv[4,6];    kv[6,5] = kv[5,6];    kv[6,6] = kt*n33;   
    
    kv = @view ke[1:6,7:12]
    for j=1:6
        for i=1:6
            kv[i,j] = -ke[i,j]
        end
    end
    kv = @view ke[7:12,1:6]
    for j=1:6
        for i=1:6
            kv[i,j] = ke[i,j]
        end
    end
    kv = @view ke[7:12,7:12]
    for j=1:6
        for i=1:6
            kv[i,j] = -ke[i,j]
        end
    end
end

function truss3d2_bmat!(ba, bt, coord)
"""
    function truss3d2_bmat!(ba, bt, coord)
        Generates the axial, and torsional B-matrices for a 3D 
        2-node truss element. 
    
     	coord = coordinates at the element ends in column format.  The 
                first column is the first node and second column holds 
                the second node.
    
       So to compute the strain at a given location on the beam cross-
       section, given by y=c
    
              strain = (ba + c*bt)*d 
    
       Note: the B-matrix for bending is linear w.r.t. xi so you only 
       need to calculate the B-matrix at the end points to capture the 
       maximum strain.
"""  
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL2 = (1/norm(n2-n1))^2
    nx = (n2[1]-n1[1])*invL2
    ny = (n2[2]-n1[2])*invL2  
    nz = (n2[3]-n1[3])*invL2  
    
    ba[1,1] = -nx;  ba[1,2] = -ny;  ba[1,3] = -nz; 
    ba[1,4] = 0.0;  ba[1,5] = 0.0;  ba[1,6] = 0.0; 
    ba[1,7] =  nx;  ba[1,8] =  ny;  ba[1,9] =  nz; 
    ba[1,10] = 0.0; ba[1,11] = 0.0; ba[1,12] = 0.0; 

    bt[1,1] = 0.0;  bt[1,2] = 0.0;  bt[1,3] = 0.0; 
    bt[1,4] = -nx;  bt[1,5] = -ny;  bt[1,6] = -nz; 
    bt[1,7] = 0.0;  bt[1,8] = 0.0;  bt[1,9] = 0.0; 
    bt[1,10] = nx;  bt[1,11] =  ny; bt[1,12] =  nz; 

end

function truss3d2_mmat!(me, coord, rhoA)
    """
        Function truss3d2_mmat!(me, coord, rhoA)
    
        Computes the consistent mass matrix for a 2 node truss element in
        3-dimensions.
            me = the 6x6 consistent mass matrix on return
            coord =  coordinates at the element ends in column format.  
                     The first column is the first node and second column 
                     holds the second node.
            rhoA =  the specific density, ie.e. the density times the 
                    cross-sectional area
    """
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    rhoAL = rhoA*norm(n2-n1)
    mii = rhoAL/3
    mij = rhoAL/6   
    fill!(me, 0.0)
    for i = 1:6  
        me[i,i] = mii
        for j = i+3:3:6
            me[i,j] = mij
            me[j.i] = mij
        end
    end 
end
#----------------------------------------------------------------------
# BEAM2D2 Element - two node beam element in 2D
function beam2d2_kmat!(ke, coord, E, I, A, wa=nothing, wb=nothing, 
                  pa=nothing, pb=nothing, GK=nothing)
"""
    function beam2d2_kmat!(ke, coord, E, I, A, wa, wb, pa, pb, GK)

     	Generates equations for a beam element in 2D
    
        ke = Returns the 6x6 element stiffness matrix 
     	coord = coordinates at the element ends in column format.  The 
                first column is the first node and second column holds 
                the second node.
     	E = modulus of elasticity
     	I = 2nd area moment that resist bending 
     	A = area of cross-section
       wa, wb = neutral axis offset at nodes a and b.  These are offset
                vectors in the golbal space of length 2.
       pa, pb = released dofs at nodes a and b.  These should be each of
                length 3 where a value of unity (1) indicates the dof 
                should be released.  This will zero out the 
                corresponding row and column.
       GK = Shear stiffness correction factor K multiplied by the shear 
            modulus G.
"""
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL = 1/norm(n2-n1)
    
    EI = E*I
    
    fill!(ke, 0.0)
    ka = A*E*invL
    ke[1,1] = ka;  ke[1,4] = -ka;
    ke[4,1] =-ka;  ke[4,4] =  ka;

    ke[2,2] = 12*E*I*invL^3; ke[2,3] = 6*E*I*invL^2;
    ke[2,5] = -ke[2,2];      ke[2,6] = ke[2,3];
    
    ke[3,2] =  ke[2,3];   ke[3,3] = 4*E*I*invL;
    ke[3,5] = -ke[3,2];   ke[3,6] = 2*E*I*invL;

    ke[5,2] = ke[2,5]; ke[5,3] = ke[3,5]; 
    ke[5,5] = ke[2,2]; ke[5,6] = ke[3,5];

    ke[6,2] = ke[2,6]; ke[6,3] = ke[3,6]; 
    ke[6,5] = ke[5,6]; ke[6,6] = ke[3,3]; 
    
    # if given add in the transverse shear factor K
    if !isnothing(GK)
        kb = ke[2,2] 
        ks = A*GK*invL
        k = kb*ks/(kb+ks)
        ke[2,2] =  k; ke[2,5] = -k;
        ke[5,2] = -k; ke[5,5] =  k;
    end
          
    # Transform ke into the computational coordinate system
    c = (n2[1]-n1[1])*invL
    s = (n2[2]-n1[2])*invL
    T = @view _ScRaTcH_REAL_[1:6,1:6]
    T[1,1] = c; T[1,2] = -s; T[2,1] = s; T[2,2] = c;
    T[4,4] = c; T[4,5] = -s; T[5,4] = s; T[5,5] = c;
    ke[1:6,1:6] = T'*ke[1:6,1:6]*T
    return
    if isnothing(wa); return; end
    if isnothing(wb); wb = wa; end
   
    # Transform ke by the offsets 
    W = @view _ScRaTcH_REAL_[1:6,1:6] 
    fill!(W,0.0)
    for i in 1:6; W[i,i]=1.0; end
    W[1,3] = -wa[2]; W[2,3] = wa[1];
    W[4,6] = -wb[2]; W[5,6] = wb[1];
    ke[1:6,1:6] .= W'*ke[1:6,1:6]*W;
    
    # zero out the rows and columns of the released dofs given in pa and pb
    if isnothing(pa); return; end
    release_dof!(ke, pa, 0)

    if isnothing(pb); return; end
    release_dof!(ke, pb, 3)
end

function beam2d2_bmat!(bb, ba, coord, xi, wa=nothing, wb=nothing)
"""
function  beam2d2_bmat!(bb, ba, coord, xi, wa, wb)
    Generates the bending and axial B-matrices for a 2D 2-node beam 
    element. 
    
    coord = coordinates at the element ends in column format.  The 
        first column is the first node and second column holds the 
        second node.
    xi = parent element coordinate along the beam, xi in [0, 1]. 
    wa, wb = neutral axis offset at nodes a and b (not yet implemented)
    
    So to compute the strain at a given location on the beam cross-
    section, given by y=c
    
        strain = (ba + c*bb)*d 
    
    Note: the B-matrix for bending is linear w.r.t. xi so you only need
    to calculate the B-matrix at the end points to capture the maximum 
    strain.
"""    
    n1 = @view coord[:,1]
    n2 = @view coord[:,2]
    invL = 1/norm(n2-n1)
    c = (n2[1]-n1[1])*invL
    s = (n2[2]-n1[2])*invL
    
    # axial B-matrix
    ba[1,1] =  -invL*c; ba[1,2] =  -invL*s; ba[1,3] = 0.0;
    ba[1,4] = -ba[1,1]; ba[1,5] = -ba[1,2]; ba[1,6] = 0.0;
    
    bv = 6*(2*xi-1)*invL^2;
    
    # bending B-martix
    bb[1,1] = -s*bv; bb[1,2] = -c*bv; bb[1,3]= -(6*xi-4)*invL;
    bb[1,4] =  s*bv; bb[1,5] =  c*bv; bb[1,6]= -(6*xi-2)*invL;

    if isnothing(wa); return; end
    if isnothing(wb); wb = wa; end

    # Transform bb by the offsets
    A = @view _ScRaTcH_REAL_[1:6,1:6] 
    fill!(A,0.0)
    for i in 1:6; A[i,i]=1.0; end
    A[1,3] = -wa[2]; A[2,3] = wa[1];
    A[4,6] = -wb[2]; A[5,6] = wb[1];
    bb .= bb*A
    ba .= ba*A    
end    
           
#----------------------------------------------------------------------
# BEAM3D2 Element - two node beam element in 3D
function offset_gridpts_3d!(A, wa, wb)
    fill!(A, 0.0)
    for i in 1:12
        A[i,i] = 1.0
    end
                        A[1,5] =  wa[3];    A[1,6] = -wa[2]; 
    A[2,4] = -wa[3];                        A[2,6] =  wa[1];
    A[3,4] =  wa[2];    A[3,5] = -wa[1];  

                        A[7,11] =  wb[3];   A[7,12] = -wb[2]; 
    A[8,10] = -wb[3];                       A[8,12] =  wb[1];
    A[9,10] =  wb[2];   A[9,11] = -wb[1];
end

# function set_transformation_matrix_3d!(T, coord)
# end
# function set_transformation_matrix_3d!(T, coord, I1, I2, I12)
# end

function beam3d2_kmat!(ke, coord, v, E, G, A, J, I1, I2, 
     I12=nothing, wa=nothing, wb=nothing, 
     pa=nothing, pb=nothing, K1=nothing, K2=nothing)
"""
    beam3d2_kmat!(ke, coord, v, E, G, A, J, I1, I2, I12, wa, wb, pa, pb, K1, K2)
    
    Generates equations for a space frame element in 3D
    
    coord = coordinates at the element ends in column format.  The first 
            column is the first node and second column holds the second node.
    v =  orientation vector to define element 1 plane
    E = modulus of elasticity
    G = shear modulus
    J = torsional rigity
    A = area of cross-section
    I1, I2 = 2nd area moment that resist bending in the element 1 and
                2 planes (Nastran convention)
   	I12 = cross product 2nd area moment
    wa, wb = neutral axis offset at nodes a and b
    pa, pb = released dofs at nodes a and b
    K1, K2 = Shear stiffness correction factors (not yet implemented)
"""
    n1 = @view coord[1:3,1]
    n2 = @view coord[1:3,2]
    invL = 1/norm(n2-n1)
    
    T = @view _ScRaTcH_REAL_[1:12,1:12]
    fill!(T, 0.0)

    e1 = @view _ScRaTcH_REAL_[1,1:3]
    e2 = @view _ScRaTcH_REAL_[2,1:3]
    e3 = @view _ScRaTcH_REAL_[3,1:3]

    # Construct the element x, y and z axis.  This is using the
    # Nastran element sign convention 
    e1 .= (n2 - n1)*invL
    e3 .= cross(e1, v)
    e3 .= e3/norm(e3)
    e2 .= cross(e3, e1)
    
    if isnothing(I12)
        EI1=E*I1
        EI2=E*I2
    else
        # find the principal inertials and the axis directions.
        thetap = 0.5*atan2(-2*I12,(I1-I2));
        R = @view _ScRaTcH_REAL_[4:6,1:3]
        R = [1 0 0; 0 cos(thetap) sin(thetap); 0 -sin(thetap) cos(thetap) ]
        e2 = e2*R
        e3 = e3*R
        r = sqrt((0.5*(I1-I2))^2 + I12^2)
        c = 0.5*(I1+I2)
        EI1 = E*(c + r)
        EI2 = E*(c - r)
    end
    for j in 1:3
        for i in 1:3
            T[i+3,j+3] = T[i,j]
            T[i+6,j+6] = T[i,j]
            T[i+9,j+9] = T[i,j]
        end
    end
    
    # construct ke in the principal coordinate system
    fill!(ke, 0.0)
    kt = G*J*invL 
    ka = E*A*invL
    ke[1,1] = ka; ke[1,7] = -ka; ke[7,1] = -ka; ke[7,7] = ka;
    ke[4,4] = kt; ke[4,10] = -kt; ke[10,4] = -kt; ke[10,10] = kt;

    k1 = 12*EI1*invL^3; k2=6*EI1*invL^2; k3=2*EI1*invL;
    ke[ 2, 2] =  k1; ke[ 6, 2] =  k2;   ke[ 8, 2] = -k1;    ke[12, 2] =  k2;
    ke[ 2, 6] =  k2; ke[ 6, 6] =2*k3;   ke[ 8, 6] = -k2;    ke[12, 6] =  k3;
    ke[ 2, 8] = -k1; ke[ 6, 8] = -k2;   ke[ 8, 8] =  k1;    ke[12, 8] = -k2;
    ke[ 2,12] =  k2; ke[ 6,12] =  k3;   ke[ 8,12] = -k2;    ke[12,12] =  2*k3;

    k1 = 12*EI2*invL^3;  k2=6*EI2*invL^2; k3=2*EI2*invL;
    ke[ 3, 3] =  k1; ke[ 5, 3] = -k2;   ke[ 9, 3] = -k1;    ke[11, 3] = -k2;
    ke[ 3, 5] = -k2; ke[ 5, 5] =2*k3;   ke[ 9, 5] =  k2;    ke[11, 5] =  k3;
    ke[ 3, 9] = -k1; ke[ 5, 9] =  k2;   ke[ 9, 9] =  k1;    ke[11, 9] =  k2;
    ke[ 3,11] = -k2; ke[ 5,11] =  k3;   ke[ 9,11] =  k2;    ke[11,11] = 2*k3;

    # Transform k matrix to global/computational system
    ke = T'*ke*T

    if isnothing(wa); return; end
    if isnothing(wb); wb = wa; end

    # Transform ke by the node/grid offsets  
    A = @view _ScRaTcH_REAL_[1:12,1:12]
    offset_gridpts_3d!(A, wa, wb)
    ke = A'*ke*A;

    # zero out the rows and columns of the released dofs given in pa and pb
    if isnothing(pa); return; end
    release_dof!(ke, pa, 0)

    if isnothing(pb); return; end
    release_dof!(ke, pb, 6)
end

function beam3d2_bmat!(bb1, bb2, ba, bt, coord, v, xi, wa, wb)
"""
function beam3d2_bmat!(bb1, bb2, ba, bt, coord, v, xi, wa, wb)
    Generates the bending, axial, and torsional B-matrices for a 3D 
    2-node beam element. 
    
    coord = coordinates at the element ends in column format.  The first 
           column is the first node and second column holds the second 
           node.
    xi = parent element coordinate along the beam, xi in [0, 1]. 
    wa, wb = neutral axis offset at nodes a and b (not yet implemented)
    
    So to compute the strain at a given location on the beam cross-
    section, given by y=c
    
        strain = (ba + c*bt + f*bb)*d 
    
    Note: the B-matrix for bending is linear w.r.t. xi so you only need
    to calculate the B-matrix at the end points to capture the maximum 
    strain.
"""
    n1 = @view coord[1:3,1]
    n2 = @view coord[1:3,2]
    invL = 1/norm(n2-n1)
    
    T = @view _ScRaTcH_REAL_[1:12,1:12]
    fill!(T, 0.0)

    e1 = @view _ScRaTcH_REAL_[1,1:3]
    e2 = @view _ScRaTcH_REAL_[2,1:3]
    e3 = @view _ScRaTcH_REAL_[3,1:3]

    # Construct the element x, y and z axis.  This is using the
    # Nastran element sign convention 
    e1 .= (n2 - n1)*invL
    e3 .= cross(e1, v)
    e3 .= e3/norm(e3)
    e2 .= cross(e3, e1)
    
    for j in 1:3
        for i in 1:3
            T[i+3,j+3] = T[i,j]
            T[i+6,j+6] = T[i,j]
            T[i+9,j+9] = T[i,j]
        end
    end
    
    fill!(ba[1,1:12],0.0); 
    ba[1,1]= -e1[1]; ba[1,2]= -e1[2]; ba[1,3]= -e1[3]; 
    ba[1,7]=  e1[1]; ba[1,8]=  e1[2]; ba[1,9]=  e1[3]; # axial B-matrix
    fill!(bt[1:12],0.0); 
    bt[1,4] = -e1[1]; bt[1,5] = -e1[2]; bt[1,5] = -e1[3]; 
    bt[1,10]=  e1[1]; bt[1,11]=  e1[2]; bt[1,12]=  e1[3]; # torsion B-matrix
    
    # bending B-martices
    bv = 6*(2*xi-1)*invL^2
    fill!(bb1[1,1:12],0.0); 
    fill!(bb2[1,1:12],0.0); 
    bb1[1,2] = -bv; bb1[1,6]  = -(6*xi-4)*invL; 
    bb1[1,8] = bv;  bb1[1,12] = -(6*xi-2)*invL;
    bb2[1,3] = -bv; bb1[1,5]  = bb1[1,6]; 
    bb2[1,9] = bv;  bb1[1,11] = bb1[1,12];

    ba .= ba*T;
    bt .= bt*T;
    bb1 .= bb1*T;
    bb2 .= bb2*T;
         
    if isnothing(wa); return; end
    if isnothing(wb); wb = wa; end

    # Transform bb by the offsets
    A = @view _ScRaTcH_REAL_[1:12,1:12]
    offset_gridpts_3d!(A, wa, wb)

    bb1 .= bb1*A;
    bb2 .= bb2*A;
    ba .= ba*A;
    bt .= bt*A;

end   

#----------------------------------------------------------------------
# QUAD2D4 Element - Four node quadrilateral element in 2D.  Depending
# on the C matrix passed, this can be plane stress or plane strain.
function quad2d4_bmat!(B, coord, xi=nothing)

    jac = 0.0
    dNdxi = @view B[1:2,1:4]
    jmat = @view B[1:2,5:6]

    if isnothing(xi)
        r = 0.0; s = 0.0
    else
        r = xi[1]; s = xi[2]
    end

    dNdxi[1,1] = -0.25*(1-s)
    dNdxi[1,2] =  0.25*(1-s)
    dNdxi[1,3] =  0.25*(1+s)
    dNdxi[1,4] = -0.25*(1+s)
    dNdxi[2,1] = -0.25*(1-r)
    dNdxi[2,2] = -0.25*(1+r)
    dNdxi[2,3] =  0.25*(1+r)
    dNdxi[2,4] =  0.25*(1-r)

    jmat = coord*dNdxi'
    jac = (jmat[1,1]*jmat[2,2]-jmat[1,2]*jmat[2,1])
    invj = 1/jac
    for I in 1:4
        B[3,2*I]   = (jmat[2,2]*dNdxi[1,I] - jmat[1,2]*dNdxi[2,I])*invj
        B[3,2*I-1] = (-jmat[2,1]*dNdxi[1,I] + jmat[1,1]*dNdxi[2,I])*invj
    end
    for I in 1:4
        B[1,2*I]   = 0.0
        B[2,2*I-1] = 0.0
        B[1,2*I-1] = B[3,2*I]
        B[2,2*I] = B[3,2*I-1]
    end

    return jac

end

function quad2d4_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false)
    if !add
        fill!(ke, REALTYPE(0))
    end
    B = @view _ScRaTcH_REAL_[1:3, 1:8]
    for q = eachindex(qwts)
        jac = quad2d4_bmat!(B, coord, qpts[:,q])
        BTCBop!(ke, B, cmat, jac*qwts[q]*thk)
    end
end

function quad2d4_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false)
    quad2d4_kmat!(ke, coord, cmat, GAUSS2D_2PT, GAUSS2D_2WT, thk, add)
end

end 
# of the module definition
