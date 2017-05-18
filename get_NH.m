function NH = get_NH(category,H,nc)
%function: get the number of each subclass according the the given input
%
%input:
%category: number of classes 
%H: category-by-1 matrix, indicating the number of subclasses for each class
%nc: category-by-1 matrix, indicationg the number of samples for each class;
%
%output:
%NH: sum(H)-by-1 matrix, indicating the number of samples in each subclass
%
%
%Note: that the data in each class have already been sorted according to
%the NN clustering method classcluster_new.m: (find the most distant two points and put them
%at two end, then look for the samples closest to those two
%respectively...)


NH=[];
for i=1:category 
    tempNH=[]; 
    if (mod(nc(i),H(i)) == 0)
        nh = nc(i)/H(i);
        for j=1:H(i)
            tempNH = [tempNH, nh];   
        end
    else
        nh = fix(nc(i)/H(i));
        if (mod(H(i),2)~=0)  % H(i) is odd
            for j=1:(H(i)-1)/2
                tempNH = [nh,tempNH,nh];    
            end
            tempNH=[tempNH(1:j),nc(i)-nh*2*j,tempNH(j+1:end)];
        else
            for j=1:(H(i)-2)/2    %H(i) is even
                tempNH = [nh,tempNH,nh];    
            end
            if (H(i)==2) j=0; end;
            tempNH=[tempNH(1:j),nh,nc(i)-nh*(2*j+1),tempNH(j+1:end)];
        end
     end
     NH=[NH,tempNH];
end