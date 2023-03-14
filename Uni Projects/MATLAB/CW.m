clear variables;

%Part 1
[X,Y,Z] = meshgrid(-4:0.1:4);
Vexact = Potential(X,Y,Z);   
    %Refer to Potential Formula
    
%Part 2
Vr = Potential(1,-0.8,3.5);
tol = 1e-5;
A = find(X==1);
B = find(abs(Y+0.8)<tol);
C = find(Z==3.5);

%Part 3
NAN = find(isnan(Vexact)==1);
INF = find(isinf(Vexact)==1);
[a,b,c] = ind2sub(size(Vexact),INF);
[d,e,f] = ind2sub(size(Vexact),NAN);
    
%Part 4'
[X2,Y2,Z2] = meshgrid(-4.05:0.1:4.05,-4.05:0.1:4.05,-4:0.1:4);
Vexact2 = Potential(X2,Y2,Z2);

%Part 5
Vapprox = DipolePoten(X2,Y2,Z2);

%Part 6
deltaV = Vexact2 - Vapprox;
surf(X2(:,:,41),Y2(:,:,41),deltaV(:,:,41));
title('Vexact2:Vapprox difference in z=0 plane');

%Part 7
[Ex, Ey, Ez] = gradient(-Vexact2);
    %E=-dV/dr

%Part 8
[X3, Y3] = meshgrid(-4.05:0.1:4.05,-4.05:0.1:4.05);
EX = Ex(:,:,41);
EY = Ey(:,:,41);
quiver(X3, Y3, EX, EY, 0.6);
xlim([-1.55 1.55]);
ylim([-1.55 1.55]);
title('Electric field in z=0 plane');
