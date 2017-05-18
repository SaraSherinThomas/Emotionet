% This is a demo which shows how the Homoscedastic criterion works. 

% This demo consists of two parts: visualization and classification. First,
% data from two mixture of Gaussians are generated as an XOR format and each
% class represents a mixture of two Gaussian distributions. Kernel Subclass
% Discriminant Analysis (KSDA) or Kernel Discriminant Analysis (KDA) is
% used to discriminate two classes. The RBF kernel parameter and the
% subclass divisions are  optimized by the Homoscedastic
% criterion. First, the discriminant function is plotted and one can 
% visualize the classification boundary.
% Then some testing samples from the same distributions are generated and
% are classified. The classification accuracy obtained by the 
% nearest neighbor classifier in the kernel subspace is shown.

% Copyrighted code
% (c) Di You, Onur Hamsici and Aleix M Martinez
%
% For additional information contact the authors


N=60; % N: umber of samples in each class
d=2; % d: dimensionality
C=2; % C; number of classes

% generate two Gaussian mixture classes
m1=[0.25, 1/4];
m2=[-0.25 -1/4];
m3=[0.25 -1/4];
m4=[-0.25 1/4];


teta=-15;
R=[cos(teta) -sin(teta); sin(teta) cos(teta)];
X1=randn(N,d)*diag([.1 .1])*R+repmat(m1,N,1);
X2=randn(N,d)*diag([.1 .1])*R+repmat(m2,N,1);
teta=15;
R=[cos(teta) -sin(teta); sin(teta) cos(teta)];
X3=randn(N,d)*diag([.1 .1])*R+repmat(m3,N,1);
X4=randn(N,d)*diag([.1 .1])*R+repmat(m4,N,1);

Xx=[X1;X2;X3;X4];
nc=[2*N,2*N]; 
% nearest neighbor clustering
Xtrain = NNclassclustering(Xx,2,nc);
trainingdata=Xtrain';
% plot the training samples
figure ;
plot(trainingdata(1,1:2*N),trainingdata(2,1:2*N),'r>','MarkerSize',15,'LineWidth',3);
%hold on
plot(trainingdata(1,2*N+1:end),trainingdata(2,2*N+1:end),'o','MarkerSize',15,'LineWidth',3);hold on
l=size(trainingdata,2);

% get the Euclidean distance matrix of the pairwise samples
A = trainingdata'*trainingdata;
dA = diag(A);
DD = repmat(dA,1,l) + repmat(dA',l,1) - 2*A;
s1=sum(sum(DD,1));
num=l*(l-1)/2;
miu_DD=s1/2/num;
% specify the quasi-Newton optimization method with a BFGS update
options = optimset('LargeScale','off', 'Display','iter', 'GradObj','off',...
'HessUpdate','bfgs', 'TolX',1e-10, 'MaxFunEvals',5000, 'MaxIter',10000);

% determine the optimal kernel parameter and subclass divisions using the
% Homoscedastic criterion if one uses KSDA. If KDA is used, only the kernel parameter
% needs to be optimized and one can set "H=ones(1,C)" to use KDA to do
% classification. 
for ii=1:5
H = ii*ones(1,C); % specify the subclass divisions in each class
NH = get_NH(C,H,nc);
X0=sqrt(miu_DD/2); % initialization for kernel parameter
[Sigma(ii),fval(ii)] = fminunc(@(sigma)Maxhomo(H, C, NH, l, sigma,DD),X0,options);
end

% select the optimal subclass numbers in each class and optimal kernel
% parameter
[F,ind]=min(fval);
op_H=ind
op_sigma=Sigma(ind)

H = op_H*ones(1,C);
NH = get_NH(C,H,nc);

K1=exp(-DD/(2*op_sigma^2));  % calculate the kernel matrix

% KSDA classification. If KDA is used, set "H=ones(1,C)" and obtain the
% corresponding optimal kernel parameter.
v=KSDA(C,trainingdata,H,NH,K1);
figure ;
% plot the classification boundary
[zx,zy] = meshgrid(-.8:.02:.6, -.8:.02:.6);
z=[zx(:),zy(:)];
z=z';

L1=1/N*ones(N,1);
mean1_proj=v'*K1(:,1:N)*L1;
mean2_proj=v'*K1(:,N+1:2*N)*L1;
mean3_proj=v'*K1(:,2*N+1:3*N)*L1;
mean4_proj=v'*K1(:,3*N+1:4*N)*L1;
mean_proj=[mean1_proj';mean2_proj';mean3_proj';mean4_proj'];
for i=1:4*N
    for j=1:size(z,2)
        dd(i,j)=norm(trainingdata(:,i)-z(:,j))^2;
    end
end
G1=exp(-dd/(2*op_sigma^2)); 
clear dd
rr=dsearchn(mean_proj, G1'*v);
for i=1:size(rr,1)
    if rr(i)==2
        rr(i)=1;
    end
end
for i=1:size(rr,1)
    if rr(i)==3 || (rr(i)==4)
        rr(i)=2;
    end
end

rxy=zeros(size(zx));
rxy(:)=rr;
colormap(gray);
pcolor(zx,zy,rxy);
shading interp
%  contour(zx,zy,rxy,[1.5 1.5])


%%%%%%%%% generate testing data
N2=100; 
Y1=randn(N2,d)*diag([.1 .1])*R+repmat(m1,N2,1);
Y2=randn(N2,d)*diag([.1 .1])*R+repmat(m2,N2,1);
Y3=randn(N2,d)*diag([.1 .1])*R+repmat(m3,N2,1);
Y4=randn(N2,d)*diag([.1 .1])*R+repmat(m4,N2,1);
Y=[Y1;Y2;Y3;Y4];
testingdata=Y';
test_label=[ones(1,2*N2),2*ones(1,2*N2)];

%  -------------- testing --------------------------

rate=KSDA_MaxHomo(trainingdata,C,nc,testingdata,test_label);
fprintf('the optimal number of subclass is %d\n', op_H)
fprintf('the optimal kernel parameter is %d\n', op_sigma)
fprintf('the classification accuracy is %d\n', rate)