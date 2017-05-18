function v=KSDA(C,X,H,nh,K1)

% Kernel Subclass Discriminant Analysis (KSDA)

% Input: 

% C: number of classes

% X: The p-by-N training data matrix, where N is the number of
% training samples and p is the number of dimensions of the data. Note that
% this matrix should be formatted as follows (suppose n_i is the number of 
% samples in ith class): The first n_1 columns are the n_1 samples from 1st
% class, the next n_2 columns are the n_2 samples from 2nd class, etc.

% H: a 1-by-C vector with each element indicating the number of subclassses
% in each class.

% nh: a 1-by-C*sum(H) vector with each element indicating the number of
% samples in each subclass.

% K1: the kernel matrix

% Output

% v: the projection matrix

% Copyrighted code
% (c) Di You, Onur Hamsici and Aleix M Martinez
%
% For additional information contact the authors





p = size(X,1);
l = size(X,2);
HH = sum(H);
Q=zeros(l,size(nh,2));
start=0;
for i=1:size(nh,2)
    
    for j=start+1:start+nh(i)
        Q(j,i)=1/nh(i);
    end
    start=sum(nh(1:i));
end

for i=1: size(nh,2)
    for class = 1:C
        if (i <= sum(H(1:class)))
            sub_label(i) = class;
            break;
        end
    end
    
end

% define kernel between subclass scatter matrix
A= zeros(l,l);
for i=1:size(nh,2)-1
    for j=i+1:size(nh,2)
        if (sub_label(i) ~= sub_label(j))
          
            A  = A +  (nh(i)/l)*(nh(j)/l)*(Q(:,i)-Q(:,j))*(Q(:,i)-Q(:,j))';
        end
    end
end

S_M=K1*A*K1;

% define kernel covariance matrix
opts.disp=0;
rankB=HH-1;
B=eye(l)-1/l^2*ones(l,l);
S_N=K1*B*K1;
S_N=S_N+0.0001*eye(size(S_N,1)); 
[vn,dn]=eig(S_N);
tem1=diag(dn);
tem1=1./tem1;
dn=diag(tem1);
invS_N=vn*dn*vn';

% solve the generalized eigenvalue decomposition to get the projection
% matrix
[v,d]=eigs(invS_N*S_M,rankB,'lm',opts);
