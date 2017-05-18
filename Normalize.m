srcFiles = dir('/media/sherin/New Volume/ML PROJECT/Project/cohn-kanade-images/S005/001/S005_001_00000001.png');
imgPath = 'E:/ML PROJECT/Shoulder Pain dataset/cohn-kanade-images/S059/001/';
landMarkPath = 'E:/ML PROJECT/Shoulder Pain dataset/Landmarks/S059/001/';
imgType = '*.png'; % change based on image type
landmarkType='*.txt';
images  = dir([imgPath imgType]);
landmarks = dir([landMarkPath landmarkType]);
imgNo=0;
auNo=0;
for idx = 1:length(images)
    Seq{idx,1} = imread([imgPath images(idx).name]);
    Seq{idx,2} = importdata([landMarkPath landmarks(idx).name]);
    lmPoints=Seq{idx,2};
    Seq{idx,3} ={ (lmPoints(37,1)+lmPoints(40,1))/2 , (lmPoints(37,2)+lmPoints(40,2))/2};
    Seq{idx,4} ={ (lmPoints(43,1)+lmPoints(46,1))/2 , (lmPoints(43,2)+lmPoints(46,2))/2};
    leftEyeCenter=Seq{idx,3};
    rightEyeCenter=Seq{idx,4};
    eyeDist=norm([abs(leftEyeCenter{1,1}-rightEyeCenter{1,1}),abs(leftEyeCenter{1,2}-rightEyeCenter{1,2})]);
    Seq{idx,5}=lmPoints*300/eyeDist;
    normLM = Seq{idx,5};
    for i = 1:68
        for j = 1:68
            if(i>j)
                distPoints(i,j) = norm([abs(normLM(i,1)-normLM(j,1)),abs(normLM(i,2)-normLM(j,2))]);
            end
        end
    end
    Seq{idx,6}=distPoints;
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
    Seq{idx,7}=angleAtLM;
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
        Xij=[Xij angleAtLM{i}];
    end
      %x(imgNo,auNo)
   
end
hold on
axis ij;
triplot(tri);
triplot(tri(ti{:},:),normLM(:,1),normLM(:,2),'Color','r') % vertex 5 (in red)
hold off;
%imshow(Seq{1,1});
leftEyeCenter=Seq{1,3};
rightEyeCenter=Seq{1,4};
%plot(leftEyeCenter{1,1},leftEyeCenter{1,2},'o','MarkerEdgeColor', 'k', 'MarkerFaceColor','green');
%plot(rightEyeCenter{1,1},rightEyeCenter{1,2},'o','MarkerEdgeColor', 'k', 'MarkerFaceColor','green');
lambda = [4,4*sqrt(2),8,8*sqrt(2),16]
gamma = 1;
alpha = [4,6,8,10];
phi = [0,1,2];