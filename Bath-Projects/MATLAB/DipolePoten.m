function [D_V] = DipolePoten(X,Y,Z)
k = (4*pi*8.85*10^-15)^-1;
Q = 1;
D_V = (k./(X.^2+Y.^2+Z.^2))*(-2).*(Y./sqrt(X.^2+Y.^2+Z.^2));
%Dipole poten V=((kQ)/r^2)*d.(r/abs(r))
end
