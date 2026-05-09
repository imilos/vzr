%
% F-ja koja racuna vrednost distribucije verovatnoce
%
function vrednost = f(x)
    if (0<=x) && (x<=pi/4)
        vrednost = sin(x);
     elseif (pi/4<=x) && (x<=2+pi/4)
        vrednost = (-4*x+pi+8)/(8*sqrt(2));
    else 
        vrednost = 0; 
    end
endfunction

i=1;
while i < 10000
    x = rand*(2+pi/4);
    u = rand;
    if u*sqrt(2)/2 <= f(x)
        y(i) = x;
        i = i + 1;
    end
end
hist(y,100)
