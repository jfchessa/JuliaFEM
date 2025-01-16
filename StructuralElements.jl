__precompile__()

module StructuralElements

using FemBasics: BTCBop!, IDTYPE, REALTYPE, 
    GAUSS2D_2PT, GAUSS2D_2WT, GAUSS2D_3PT, GAUSS2D_3WT, 
    GAUSS3D_2PT, GAUSS3D_2WT, GAUSS3D_3PT, GAUSS3D_3WT, 
    SIMPLEX2D_1PT, SIMPLEX2D_1WT, SIMPLEX2D_3PT, SIMPLEX2D_3WT,
    SIMPLEX3D_4PT, SIMPLEX3D_4WT, SIMPLEX3D_5PT, SIMPLEX3D_5WT, 
    dshape_tria3!, dshape_tria6!, dshape_quad4!, dshape_quad8!,
    dshape_tetra4!, dshape_tetra10!, dshape_hexa8!, dshape_hexa20!

using LinearAlgebra
using StaticArrays

export truss2d2_kmat!, truss2d2_bmat!, beam2d2_kmat!, beam2d2_bmat!
export truss2d2_mmat!, truss3d2_mmat!
export truss3d2_kmat!, truss3d2_bmat!, beam3d2_kmat!, beam3d2_bmat!
export tria3_bmat!, tria3_kmat!, tria6_bmat!, tria6_kmat! 
export quad4_bmat!, quad4_kmat!, quad8_bmat!, quad8_kmat!
export tetra4_bmat!, tetra4_kmat!, tetra10_bmat!, tetra10_kmat!
export hexa8_bmat!, hexa8_kmat!, hexa20_bmat!, hexa20_kmat!
export mat1_cmat

# This is globally allocated space for doing local calculations
# I know this is not a great idea, but does really speed things up
# Use views to help address this memory in the functions.
const global _ScRaTcH_REAL_      = zeros(REALTYPE,96,96)
const global _BMAT_SPACE_        = zeros(REALTYPE,12,96)
const global _SHAPE_FUNCT_SPACE_ = zeros(REALTYPE,60,6)

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

# *********************************************************************
#
#           T R U S S    A N D     B E A M     E L E M E N T S
#
# *********************************************************************
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

# *********************************************************************
#
#                      S H E L L     E L E M E N T S
#
# *********************************************************************

# *********************************************************************
#
#            2 D     C O N T I N U U M    E L E M E N T S
#
# *********************************************************************

#----------------------------------------------------------------------
# Generic routines for 2D Continuum Elements
function gradshape2d!(dN, coord, nn::Int)
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

function fillB2D!(B, dNdx, nn::Int=size(dNdx,1)) 
    for i in 1:nn
        B[1,2*i-1] = dNdx[i,1]; B[1,2*i] = 0.0; 
        B[2,2*i-1] = 0.0;       B[2,2*i] = dNdx[i,2]; 
        B[3,2*i-1] = dNdx[i,2]; B[3,2*i] = dNdx[i,1]; 
    end
end

function gelem2dc_bmat!(B, coord, xi, dshapefunc, nn::Int=size(coord,2))
    dN = @view _SHAPE_FUNCT_SPACE_[1:nn,1:2]  
    dshapefunc(dN, xi)
    jac = gradshape2d!(dN, coord, nn)
    fillB2D!(B, dN, nn) 
    return jac
end

function gelem2dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, bmatfunct, nn::Int=size(coord,2))
    if !add
        fill!(ke[2:nn,2:nn], REALTYPE(0))
    end
    B = @view _BMAT_SPACE_[1:3, 1:2*nn]
    for q = eachindex(qwts)
        jac = bmatfunct(B, coord, qpts[q])
        BTCBop!(ke, B, cmat, jac*qwts[q]*thk)
    end
end

#----------------------------------------------------------------------
# TRIA3 Element 
function gradshapeCST!(dN, coord)
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

function tria3_bmat!(B, coord, xi=nothing) 
    dN = @view _SHAPE_FUNCT_SPACE_[1:3,1:2]  
    jac = gradshapeCST!(dN, coord)
    fillB2D!(B, dN, 3) 
    return jac
end

function tria3_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) 
    if !add
        fill!(ke[1:6,1:6], REALTYPE(0))
    end
    B = @view _BMAT_SPACE_[1:3, 1:6]
    jac = tria3_bmat!(B, coord)
    BTCBop!(ke, B, cmat, 0.5*jac*thk)
end

#----------------------------------------------------------------------
# TRIA6 Element
tria6_bmat!(B, coord, xi) = gelem2dc_bmat!(B, coord, xi, dshape_tria6!, 6)

tria6_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem2dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, tria6_bmat!, 6)

tria6_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    tria6_kmat!(ke, coord, cmat, SIMPLEX2D_3PT, SIMPLEX2D_3WT, thk, add)

#----------------------------------------------------------------------
# QUAD4 Element - Four node quadrilateral element in 2D.  Depending
# on the C matrix passed, this can be plane stress or plane strain.
quad4_bmat!(B, coord, xi) = gelem2dc_bmat!(B, coord, xi, dshape_quad4!, 4)

quad4_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem2dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, quad4_bmat!, 4)

quad4_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    quad4_kmat!(ke, coord, cmat, GAUSS2D_2PT, GAUSS2D_2WT, thk, add)

#----------------------------------------------------------------------
# QUAD8 Element
quad8_bmat!(B, coord, xi) = gelem2dc_bmat!(B, coord, xi, dshape_quad8!, 8)

quad8_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem2dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, quad8_bmat!, 8)

quad8_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    quad8_kmat!(ke, coord, cmat, GAUSS2D_3PT, GAUSS2D_3WT, thk, add)

# *********************************************************************
#
#            3 D     C O N T I N U U M    E L E M E N T S 
#
# *********************************************************************

#----------------------------------------------------------------------
# Generic routines for 3D Continuum Elements
function gradshape3d!(dN, coord)
    jmat = @view _ScRaTcH_REAL_[1:3,1:3]    # THIS IS NOT OPTIMIZED
    jmat = coord*dN                         # BUT NOT SURE WHAT TO DO
    dN .= dN/jmat
    return det(jmat)
end

function fillB3D!(B, dNdx, nn::Int=size(dNdx,1)) 
    for i in 1:nn
        B[1,3*i-2] = dNdx[i,1]; B[1,3*i-1] = 0.0;       B[1,3*i] = 0.0;
        B[2,3*i-2] = 0.0;       B[2,3*i-1] = dNdx[i,2]; B[2,3*i] = 0.0;
        B[3,3*i-2] = 0.0;       B[3,3*i-1] = 0.0;       B[3,3*i] = dNdx[i,3];
        B[4,3*i-2] = dNdx[i,2]; B[4,3*i-1] = dNdx[i,1]; B[4,3*i] = 0.0; 
        B[5,3*i-2] = 0.0;       B[5,3*i-1] = dNdx[i,3]; B[5,3*i] = dNdx[i,2]; 
        B[6,3*i-2] = dNdx[i,3]; B[6,3*i-1] = 0.0;       B[6,3*i] = dNdx[i,1]; 
    end
end

function gelem3dc_bmat!(B, coord, xi, dshapefunc, nn::Int=size(coord,2))
    dN = @view _SHAPE_FUNCT_SPACE_[1:nn,1:3]  
    dshapefunc(dN, xi)
    jac = gradshape3d!(dN, coord)
    fillB3D!(B, dN, nn) 
    return jac
end

function gelem3dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, bmatfunct, nn::Int=size(coord,2))
    if !add
        fill!(ke[3:nn,3:nn], REALTYPE(0))
    end
    B = @view _BMAT_SPACE_[1:6, 1:3*nn]
    for q = eachindex(qwts)
        jac = bmatfunct(B, coord, qpts[q])
        BTCBop!(ke, B, cmat, jac*qwts[q]*thk)
    end
end

#----------------------------------------------------------------------
# TETRA4 Element
function gradshapeCSTet!(dN, coord)
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

function tetra4_bmat!(B, coord, xi=nothing) 
    dN = @view _SHAPE_FUNCT_SPACE_[1:4,1:3]  
    jac = gradshapeCSTet!(dN, coord)
    fillB3D!(B, dN, 4) 
    return jac
end

function tetra4_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) 
    if !add
        fill!(ke[1:12,1:12], REALTYPE(0))
    end
    B = @view _BMAT_SPACE_[1:6, 1:12]
    jac = tetra4_bmat!(B, coord)
    BTCBop!(ke, B, cmat, 0.166666666666666667*jac*thk)
end

#----------------------------------------------------------------------
# TETRA10 Element
tetra10_bmat!(B, coord, xi) = gelem3dc_bmat!(B, coord, xi, dshape_tetra10!, 10)

tetra10_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem3dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, tetra10_bmat!, 10)

tetra10_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    tetra10_kmat!(ke, coord, cmat, SIMPLEX3D_4PT, SIMPLEX3D_4WT, thk, add)

#----------------------------------------------------------------------
# HEXA3D8 Element
hexa8_bmat!(B, coord, xi) = gelem3dc_bmat!(B, coord, xi, dshape_hexa8!, 8)

hexa8_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem3dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, hexa8_bmat!, 8)

hexa8_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    hexa8_kmat!(ke, coord, cmat, GAUSS3D_2PT, GAUSS3D_2WT, thk, add)

#----------------------------------------------------------------------
# HEXA20  Element
hexa20_bmat!(B, coord, xi) = gelem3dc_bmat!(B, coord, xi, dshape_hexa20!, 20)

hexa20_kmat!(ke, coord, cmat, qpts, qwts, thk::REALTYPE=1.0, add::Bool=false) =
    gelem3dc_kmat!(ke, coord, cmat, qpts, qwts, thk, add, hexa20_bmat!, 20)

hexa20_kmat!(ke, coord, cmat, thk::REALTYPE=1.0, add::Bool=false) =
    hexa20_kmat!(ke, coord, cmat, GAUSS3D_3PT, GAUSS3D_3WT, thk, add)

#----------------------------------------------------------------------
# PENTA3D6  Element

# ********************************************************************    
end 
# ******************** of the module definition **********************
