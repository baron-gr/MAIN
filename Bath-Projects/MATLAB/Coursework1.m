clear variables;

x = (20-(3-7*exp(2+1i*pi*cos(pi/3)))); %1
v1 = transpose(1:1:10); %2
i_comp = imag((x+(v1(5).*(2+1i)))); %3
v2 = v1.^2; %4
dot_prod = sum(v1.*v2); %5
M1 = randi([2 10], 15, 10); %6
determin = det(M1(1:10, :)); %7
Summation = sum(sum(M1>5)); %8
Mpaximum = max(M1(:, 3)); %9
plot(sin(0:pi/100:2*pi),'r'); %10
M2 = M1(1:10, :) + transpose(M1(1:10, :)); %11
max_eig = max(abs(eig(M2))); %12
int = trapz(2:0.001:5, (1+(2:0.001:5)).^2); %13
