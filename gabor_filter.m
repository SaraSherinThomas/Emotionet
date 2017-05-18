function features = gabor_filter(imgPath,lmPath)
    features=[];
    I=imread(imgPath);
    LMpoints=importdata(lmPath);
    [N N]=size(I);
    
    if size(I, 3) > 1
        I = rgb2gray(I);
    end
    I=im2double(I);
    psi=[0,1,2];
    gamma=1;
    n1=5;
    lambda=[4,4*sqrt(2),8,8*sqrt(2),16];
    n2=4;
    theta=[4,6,8,10];
    for point=1:68
        for i=1:n1
            l=lambda(i);
            sigma=[l/4,l/2,3*l/4,l];
            for j=1:n2
                t=theta(j);
                for k=1:4
                    s=sigma(k);
                    for m=1:3
                        p=psi(m);
                        x=LMpoints(point,1);
                        y=LMpoints(point,2);
                        x_theta=x*cos(t)+y*sin(t);
                        y_theta=-x*sin(t)+y*cos(t);
                        gb=exp(-.5*(x_theta.^2/s^2+y_theta.^2/s^2)).*cos(2*(pi/l)*x_theta+p);
                        features(1,end+1) = convn(gb,I,'same');
                        
                        
                    end
                end
            end
        end 
    end
    
   
    
    
end