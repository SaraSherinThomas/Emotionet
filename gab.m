
I = importdata('S005_001_00000001_landmarks.txt');
lambda=4;
alpha=10;
sigma=1;
psi=2;
gamma=1;

for i = 1:68
    x=I(i,1);
    y=I(i,2);
    
    x_alpha=x*cosd(alpha)+y*sind(alpha);
    y_alpha=-x*sind(alpha)+y*cosd(alpha);
    i1=((x_alpha^2)+(gamma^2)*(y_alpha^2))/((sigma^2)*0.5)
    i2=(cosd((2*pi*x_alpha)/lambda)+psi)
    g=exp(4)
    4.740692804800627e+05
    gb(i)= exp(-((x_alpha^2/sigma^2+y_alpha.^2/sigma^2)/2))*cosd(2*pi/lambda*x_alpha+psi);
end
imshow(gb);

