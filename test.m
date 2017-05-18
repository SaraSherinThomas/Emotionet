image_fl=getAllFiles('E:/ML PROJECT/Shoulder Pain dataset/cohn-kanade-images/');

imgNo=0;
auNo=0;
imageNo=1;

for idx = 1:4000
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
            
            %gij(imageNo,:)=gabor_filter(imgPath,lmPath);
            
            
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
    Zij(q,:)=[XijAll(q,:) angleVect];
    %Zij(q,:)=[XijAll(q,:) angleVect gij(q,:)];
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