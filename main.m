image_fl=getAllFiles('E:/ML PROJECT/Shoulder Pain dataset/cohn-kanade-images/');

imgNo=0;
auNo=0;
imageNo=1;

for idx = 1:1500
    lmPath=[];
    lmPath='E:\ML PROJECT\Shoulder Pain dataset\Landmarks';
    facsPath='E:\ML PROJECT\Shoulder Pain dataset\FACS';
    imgPath = image_fl{idx};
    [imgPathstr,imgName,imgExt] = fileparts(imgPath) ;
    facsName=strcat(imgName,'_facs.txt');
    name=strcat(imgName,'_landmarks.txt');
    path = strsplit(imgPathstr,'cohn-kanade-images');
    facsPath=strcat(facsPath,path{2},'\',facsName);
    lmPath=strcat(lmPath,path{2},'\',name);
    if(imgExt=='.png')
        if exist(facsPath, 'file') == 2
            Seq{imageNo,1} = imread(imgPath);
            
            auList=importdata(facsPath);
            Seq{imageNo,8}=auList;
        
            %lmPath=strcat(lmPath,path(2),'\',name);
            Seq{imageNo,2} = importdata(lmPath);
            lmPoints=Seq{imageNo,2};
            Seq{imageNo,3} ={ (lmPoints(37,1)+lmPoints(40,1))/2 , (lmPoints(37,2)+lmPoints(40,2))/2};
            Seq{imageNo,4} ={ (lmPoints(43,1)+lmPoints(46,1))/2 , (lmPoints(43,2)+lmPoints(46,2))/2};
            leftEyeCenter=Seq{imageNo,3};
            rightEyeCenter=Seq{imageNo,4};
            eyeDist=norm([abs(leftEyeCenter{1,1}-rightEyeCenter{1,1}),abs(leftEyeCenter{1,2}-rightEyeCenter{1,2})]);
            Seq{imageNo,5}=lmPoints*300/eyeDist;
            normLM = Seq{imageNo,5};
            for i = 1:68
                for j = 1:68
                    if(i>j)
                        distPoints(i,j) = norm([abs(normLM(i,1)-normLM(j,1)),abs(normLM(i,2)-normLM(j,2))]);
                    end
                end
            end
            Seq{imageNo,6}=distPoints;
            tri = delaunayTriangulation(normLM(:,1),normLM(:,2));
            
            %triplot(tri);
            p = tri.Points;
            triangles = tri.ConnectivityList; 
           
            for i = 1:68
                angleAtLM{i,1}=[];
                ti = vertexAttachments(tri,i);
                x1=p(i,1);y1=p(i,2);
                attachedTri=ti{1};
                for j = 1:length(attachedTri)
                    for k = 1:length(triangles)
                        if(attachedTri(j) == k)
                            TI=triangles(k,:);
                            if(triangles(k,1)==i)
                                x2 = p(triangles(k,2),1);
                                y2 = p(triangles(k,2),2);
                                x3 = p(triangles(k,3),1);
                                y3 = p(triangles(k,3),2);
                            elseif(triangles(k,2)==i)
                                x2 = p(triangles(k,1),1);
                                y2 = p(triangles(k,1),2);
                                x3 = p(triangles(k,3),1);
                                y3 = p(triangles(k,3),2);
                            elseif(triangles(k,3)==i)
                                x2 = p(triangles(k,2),1);
                                y2 = p(triangles(k,2),2);
                                x3 = p(triangles(k,1),1);
                                y3 = p(triangles(k,1),2);
                            end
                            %Area = polyarea([x1,x2,x3],[y1,y2,y3])
                            A1 = atan2(abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)),(x2-x1)*(x3-x1)+(y2-y1)*(y3-y1));
                            A1 = A1*180/pi ; 
                            P1 = [x1,y1];P2=[x2,y2];P3=[x3,y3];
    
                            angleAtLM{i,1}=[angleAtLM{i,1} A1];
                            %a1 = atan2(2*Area,dot(P2-P1,P3)-P1))
    
                        end
                    end
                end
    
            end
            Seq{imageNo,7}=angleAtLM;
            
            
            %angleAtLM(:,:)=[];
            Xij=[];
            j=0;
    
            for i=1:68
                for k=1:67
                    if(distPoints(i,k)~=0)
                        Xij=[Xij distPoints(i,k)];
                        j=j+1;
                    end
                end
            end
            
            gij(imageNo,:)=gabor_filter(imgPath,lmPath);
            
            
            imageNo
            XijAll(imageNo,:)=Xij;
            imageNo=imageNo+1;
        end
        Xij=[];
    end
    Xij=[];
    
    
end

for p=1 : 68
        for q=1:imageNo-1
            a{q,1}=Seq{q,7}{p,1};
        end
        sz = cellfun(@(x)size(x,2), a);
        minLength = min(sz);
        b = cell2mat(cellfun(@(x)x(1:minLength), a, 'uniformoutput', false))';
        b=b';
        for q=1:imageNo-1
            Seq{q,7}{p,1}=[];
            Seq{q,7}{p,1}=b(q,:);
        end
end
for q=1:imageNo-1
    angleVect=[];
    for p=1 : 68
        angleVect=[angleVect Seq{q,7}{p,1}];
    end
    %Zij(q,:)=[XijAll(q,:) angleVect];
    Zij(q,:)=[XijAll(q,:) angleVect gij(q,:)];
end

subplot(2,2,1),imshow(imread(imgPath));

subplot(2,2,2),hold on
subplot(2,2,2),axis ij;
subplot(2,2,2),triplot(tri);
subplot(2,2,2),triplot(tri(ti{:},:),normLM(:,1),normLM(:,2),'Color','r') % vertex 5 (in red)
subplot(2,2,2),hold off;
trainLabel=[];

facs2DArray=zeros(43,6);
facs=[];
sortedZij=[];
C=0;
maxAu=0;
maxInt=0;
classSamples=zeros(43,1);
for auNo=1:43
    for intensityNo=0:5
        for imageNo=1:size(Seq)
            temp=Seq{imageNo,8};
            for auPresent=1:size(temp)
                if temp(auPresent,1)==auNo
                    if temp(auPresent,2)==intensityNo
                        sortedZij=[sortedZij;Zij(imageNo,:)];
                        facs2DArray(auNo,intensityNo+1)=facs2DArray(auNo,intensityNo+1)+1;
                        classSamples(auNo,1)=classSamples(auNo,1)+1;
                        
                        if temp(auPresent,1)>maxAu
                            maxAu=temp(auPresent,1);
                        end
                        if temp(auPresent,2)>maxInt
                            maxInt=temp(auPresent,2);
                        end
                    end
                end
            end
        end
    end
    
    facs=[facs facs2DArray(auNo,:)];
    
end

d=1;

temp=facs;
facs=nonzeros(temp);
temp=[];
temp=nonzeros(classSamples);
classSamples=[];
classSamples=temp;
temp=size(facs);
C=temp(1);
sortedData = NNclassclustering(sortedZij,C,facs);
trainingdata=sortedData';
l=size(trainingdata,2);
k=1;

trainLabel=[];
for r=1 : size(facs)
    for s=1 :facs(r)
       trainLabel(k)=r;
       k=k+1;
   end
end
d=1;
lbl=[];
for r=1:43
    for s=0:5
        if facs2DArray(r,s+1)>0
            lbl(d)=r+s/10;
            d=d+1;
        end
    end
end
[trainInd,valInd,testInd] = dividerand(l,0.85,0,0.15);
limit=size(testInd);
for p=1:limit(2)
    test(:,p)=trainingdata(:,p);
    testLabel(p)=trainLabel(testInd(p));
end
v=1;
dividedTrain=[];
dividedLabel=[];
for x=1:size(trainingdata,2)
    for y = 1 : limit(2)
        if(x==testLabel(y))
            temp=0;
            for s = 1 : size(facs)
                temp=temp+facs(s);
                if(temp>=x && facs(s)>0)
                    facs(s)=facs(s)-1;
                    if(facs(s)==0)
                        lbl(s)=0;
                    end
                    break;
                end
            end
            break;
        end
    end
    if(x~=testLabel(y))
        dividedTrain(:,v)=trainingdata(:,x);
        dividedLabel(v) = trainLabel(x);
        v=v+1;
    end
end
trainingdata=[];
trainingdata=dividedTrain;
trainLabel=[];
trainLabel=dividedLabel;
temp=facs;
facs=nonzeros(temp);
temp=size(facs);
C=temp(1);
l=size(trainingdata,2);
A = trainingdata'*trainingdata;
dA = diag(A);
DD = repmat(dA,1,l) + repmat(dA',l,1) - 2*A;
s1=sum(sum(DD,1));
num=l*(l-1)/2;
miu_DD=s1/2/num;
options = optimset('LargeScale','off', 'Display','iter', 'GradObj','off',...
'HessUpdate','bfgs', 'TolX',1e-10, 'MaxFunEvals',5000, 'MaxIter',10000);
X0=sqrt(miu_DD/2); 

for ii=1:3
H = ii*ones(C,1); % specify the subclass divisions in each class
NH = get_NH(C,H,facs);
X0=sqrt(miu_DD/2); % initialization for kernel parameter
cv=ii
[Sigma(ii),fval(ii)] = fminunc(@(sigma)Maxhomo(H, C(1), NH, l, sigma,DD),X0,options);
[F,ind]=min(fval);
op_H=ind
op_sigma=Sigma(ind)
end
H = op_H*ones(C,1);
NH = get_NH(C,H,facs);
K1=exp(-DD/(2*op_sigma^2));  % calculate the kernel matrix
v=KSDA(C,trainingdata,H,NH,K1);



%[classes,rec,rate]=KSDA_MaxHomo(trainingdata,C,facs,test,testLabel);
 train=v'*K1;
 nXtest=size(test,2);
 for i=1:nXtest
    B=trainingdata-repmat(test(:,i),1,l);
    B=B.^2;
    dd(i,:)=sum(B,1);
end
dd=dd';
K2=exp(-dd/(2*op_sigma^2));
test=v'*K2;

[classes,rec,rate]=NearestNeighbor(train',test',testLabel,C,facs,trainLabel);

 
 
fprintf('the optimal number of subclass is %d\n', op_H)
fprintf('the optimal kernel parameter is %d\n', op_sigma)
fprintf('the classification accuracy is %d\n', rate)