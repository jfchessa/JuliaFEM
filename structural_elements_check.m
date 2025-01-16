% Validation for Julia StructuralElements module functions
%
% This uses femlab from jfchessa@github
clear

E = 10e6;
nu = .3;
thk = 0.25;

% check  gradshap2d!
coord = [0.0 0.0; 2.0 0.2; 0.3 1.0];
dN = [-1.0 -1.0; 1.0 0.0; 0.0 1.0];
[dNx,jac] = grad_shapefunct(coord, dN, 2, 2)

coord = [0.0 0.0; 2.0 0.2; 2.1 1.5; -.1 1.2];
xi = [.1, .2];
dN = dshape_quad4(xi)
[dNx,jac] = grad_shapefunct(coord, dN, 2, 2)

% TRIA3
coord = [0.0 0.0; 2.0 0.2; 0.3 1.0];
be = bmat_tria3(coord)
ke = kmat_tria3(coord, E, nu, thk)

% TRIA6
coord = [0.0 0.0; 2.0 0.2; 0.3 1.0; 1.0 .99; 1.16 0.6; 0.16 0.5];
xi = [0.2, 0.3];
[be, j] = bmat_tria6(coord, xi)
ke = kmat_tria6(coord, E, nu, thk)

% QUAD4
coord = [0.0 0.0; 2.0 0.2; 2.1 0.9; 0.3 1.0];
xi = [0.2, 0.3];
[be, j] = bmat_quad4(coord, xi)
ke = kmat_quad4(coord, E, nu, thk)

% QUAD8
coord = [0.0 0.0; 2.0 0.2; 2.1 0.9; 0.3 1.0;
         1.0 .1; 2.05 0.55; 1.2 1.0; 0.17 0.5 ];
[be, j] = bmat_quad8(coord, xi)
ke = kmat_quad8(coord, E, nu, thk)

% TETRA4
coord = [0.0 0.0 0.0; 2.0 0.1 0.1; 0.2 1.0 0.0; 0.0 0.1 1.0];
[be, V] = bmat_tetra4(coord);
j = 6*V
ke = kmat_tetra4(coord, E, nu);

% TETRA10
coord = [  0    1.0000         0         0    0.5000    0.5000         0         0    0.5000    0.0;     
           0         0    1.0000         0         0    0.5000    0.5000         0         0    0.5000;
           0         0         0    1.0000         0         0         0    0.5000    0.5000    0.5000 ]' ;
xi = [.09, .1, .2];
[be, j] = bmat_tetra10(coord, xi);
bbe = be([1,2,3,6,4,5],:);
ke = kmat_tetra10(coord, E, nu);

% HEXA8
coord = [0 0 0;2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2];
xi = [.09, .1, .2];
[be, j] = bmat_hexa8(coord, xi);
bbe = be([1,2,3,6,4,5],:);
ke = kmat_hexa8(coord, E, nu);

% HEXA20
coord = [0 0 0; 2 0 0; 2 2 0; 0 2 0; 0 0 2; 2 0 2; 2 2 2; 0 2 2;
         0 1 0; 2 1 0; 1 2 0; 0 1 0; 0 1 2; 2 1 2; 1 2 2; 0 1 2;
         0 0 1; 2 0 1; 2 2 1; 0 2 1];
[be, j] = bmat_hexa20(coord, xi);
bbe = be([1,2,3,6,4,5],:);
ke = kmat_hexa20(coord, E, nu);