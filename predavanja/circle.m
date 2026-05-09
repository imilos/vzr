clear all
close all

Nmax=4000;
R=5;

for n=1:Nmax
    
    %wrong method
    r1(n)=R*rand(1,1);
    theta1(n)=2*pi*rand(1,1);
    
    % right method
    % pdf_r(r)=(2/R^2) * r
    % cumulative pdf_r is F_r = (2/R^2)* (r^2)/2
    % inverse cumulative pdf is r = R*sqrt(F_r)
    % so we generate the correct r as
    r2(n) = R*sqrt(rand(1,1));
    % and theta as before:
    theta2(n)=2*pi*rand(1,1);
   
    % convert to cartesian
    x1(n)=r1(n)*cos(theta1(n));
    y1(n)=r1(n)*sin(theta1(n));
    x2(n)=r2(n)*cos(theta2(n));
    y2(n)=r2(n)*sin(theta2(n));
end

subplot(1,2,1)
plot(x1,y1,'r.')
axis([-1.1*R 1.1*R -1.1*R 1.1*R])
axis square
title('Wrong')
subplot(1,2,2)
plot(x2,y2,'g.')
axis([-1.1*R 1.1*R -1.1*R 1.1*R])
axis square
title('Right')
