function sortedtrain = NNclassclustering(trainingdata, C, nc)
% first step: find two samples (element 1 and element 2) in each class
% which have the largest distance between each other
%
% second step: sort the data such that: element 1 and element 2 are the
% 1st and nth sample in the sorted training data, and 1~n/2 samples are near
% element 1 and n/2+1~n samples are near element2 

% input: trainingdata: n-by-p matrix, all the data
%        C: number of classes
%        nc: c-by-1 matrix containing the number of samples for each class
% output:
%        trainingdata: the sorted training data


[n,p]=size(trainingdata);

 
element1=zeros(C,1);   %record the index of the two most distant samples in each class
element2=zeros(C,1);
dist=0;      %record the distance
larg_dist=zeros(1,C);%record the largest distance of every class 10

start=0;
for k=1:C
    %k   
    
    X = trainingdata(start+1:start+nc(k),:);
    s=size(X);
    if s(1)==1
        element1(k)=start+1;
        element2(k)=start+1;
    end
    for i=1:nc(k)-1
        Y = X;
        Y(1:i,:)=[];
        dist = sum((( repmat(X(i,:),nc(k)-i, 1) - Y).^2)');  
        [dist,ind] = sort(dist);
        
        if (dist(end) > larg_dist(k))
                larg_dist(k)=dist(end);
                element1(k)=i+start;
                element2(k)=ind(end)+start+i;
        end
    end    
    start=sum(nc(1:k));
end
clear X
clear Y;
% 
% filename='LargestDistanceElement'
% save(filename,'element1','element2','larg_dist');

%sorted data
start=0;
sortedtrain=zeros(n,p);
for k=1:C
    %k
    
    key1=trainingdata(element1(k),:);   % in class K, the key element 
    key2=trainingdata(element2(k),:);    
    
    sortedtrain(start+1,:)= key1; % the first and last elements in sorted trainindata are the keys
    sortedtrain(start+nc(k),:) = key2;
    
    
    num1=element1(k)-start;%(k-1)*nc;
    num2=element2(k)-start;%(k-1)*nc;
    
    temp = trainingdata(start+1:start+nc(k),:);
    s=size(temp);
    if s(1)>1
        temp(num2,:)=[];   % note: delete the second, then the first ! 
        temp(num1,:)=[];
   
        count=nc(k)-2; %198;
        for i=1:fix((nc(k)-2)/2) %74%99
    
            dist = sum(((repmat(key1,count,1) - temp ).^2)');
            [c,I]=min(dist);
            sortedtrain(start+1+i,:)=temp(I,:); %the second element in the sorted trainingdata is the one neast with key
            temp(I,:)=[]; % delete founde the nearest element
            count=count-1;
            clear dist
        
                
            dist = sum(((repmat(key2,count,1) - temp ).^2)');
            [c,I]=min(dist);
            sortedtrain(start+nc(k)-i,:)=temp(I,:); %the second element in the sorted trainingdata is the one neast with key
            temp(I,:)=[]; % delete founde the nearest element
            count=count-1;
            clear dist
    
        end
    
        if (mod(nc(k),2)~=0)
            sortedtrain(start+fix(nc(k)/2)+1,:)=temp;
        end
    else
        sortedtrain(start+1,:)=temp;
        temp=[];
    end
    start = sum(nc(1:k));
   
end


