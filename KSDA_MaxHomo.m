function [classes,rec,rate]=KSDA_MaxHomo(trainingdata,C,nc,testingdata,test_label)

% This function is an implementation of Kernel Subclass Discriminant
% Analysis (KSDA) with parameters optimized by the Homoscedastic criterion
% proposed in the paper "Kernel Optimization in Discriminant Analysis".

% Input: 

% trainingdata: The p-by-N training data matrix, where N is the number of
% training samples and p is the number of dimensions of the data. Note that
% this matrix should be formatted as follows (suppose n_i is the number of 
% samples in ith class): The first n_1 columns are the n_1 samples from 1st
% class, the next n_2 columns are the n_2 samples from 2nd class, etc.

% C: number of classes

% nc: 1-by-C vector, indicationg the number of samples in each class
% for the training data.

% testingdata: the p-by-n testing data matrix, where n is the number of
% testing samples.

% test_label: the labels for the testing data. 

% Output:

% rate: classfication accuracy


% Copyrighted code
% (c) Di You, Onur Hamsici and Aleix M Martinez
%
% For additional information contact the authors




          
%------------------------------- training stage -------------------------

          %%% Nearest Neighbor clustering of the data

          Ytrain = NNclassclustering(trainingdata',C,nc);
          trainingdata=Ytrain';
          l=size(trainingdata,2);
         
         %%% get pairwise distance matrix
          
         A = trainingdata'*trainingdata;
         dA = diag(A);
         DD = repmat(dA,1,l) + repmat(dA',l,1) - 2*A;
         
         s1=sum(sum(DD,1));
         num=l*(l-1)/2;
         mean_DD=s1/2/num;
         options = optimset('LargeScale','off', 'GradObj','off',...
            'HessUpdate','bfgs', 'TolX',1e-10, 'MaxFunEvals',5000, 'MaxIter',10000);

        %%% optimize the kernel parameter and number of subclasses in KSDA
        
        for ii=1:5
            H = ii*ones(1,C);
            NH = get_NH(C,H,nc);
            X0=sqrt(mean_DD/2);
            [Sigma(ii),fval(ii)] = fminunc(@(sigma)Maxhomo(H, C, NH, l, sigma,DD),X0,options);
        end
        [F,ind]=min(fval);
        op_H=ind;
        op_sigma=Sigma(ind);
      
        %%% perform KSDA after selecting optimal parameters
        
        H = op_H*ones(1,C);
         NH = get_NH(C,H,nc);
         K1=exp(-DD/(2*op_sigma^2));
         v=KSDA(C,trainingdata,H,NH,K1);
         
  % ---------------------- testing stage -------------------------------
         train=v'*K1;
         nXtest=size(testingdata,2);
        for i=1:nXtest
         B=trainingdata-repmat(testingdata(:,i),1,l);
         B=B.^2;
         dd(i,:)=sum(B,1);
        end
        dd=dd';
         K2=exp(-dd/(2*op_sigma^2));

        test=v'*K2;
        
        %%% nearest mean classifier
        
%         rate=Nmean(train', test', H, C, NH, test_label) 

        %%% nearest neighbor classifier
        
        [classes,rec,rate]=NearestNeighbor(train',test',test_label,C,nc);
