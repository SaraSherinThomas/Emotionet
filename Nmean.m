function rate=Nmean(X, test, H, C, nh, correct_class)

% This function performs nearest mean classifier

% Input:

% X: The N-by-p training data matrix, where N is the number of
% training samples and p is the number of dimensions of the data. Note that
% this matrix should be formatted as follows (suppose n_i is the number of 
% samples in ith class): The first n_1 columns are the n_1 samples from 1st
% class, the next n_2 columns are the n_2 samples from 2nd class, etc.

% test: the n-by-p testing data matrix, where n is the number of testing
% samples

% C: number of classes

% H: a 1-by-C vector with each element indicating the number of subclassses
% in each class.

% nh: a 1-by-C*sum(H) vector with each element indicating the number of
% samples in each subclass.

% correct_class: labels for the testing data

% Output: classification accuracy

% Copyrighted code
% (c) Di You, Onur Hamsici and Aleix M Martinez
%
% For additional information contact the authors

n = size(X,1); % number of samples
p = size(X,2); % number of dimensions

HH = sum(H);
start = 0; 
Ntest=size(test,1);
% get the mean of each subclass
meanx = mean(X);
for i=1: HH
    temp = X(start+1:start+nh(i),:);
    
    %subclass mean
    slicemean(i,:)=mean(temp,1); 

    %class label for this subclass
    for class = 1:C
        if (i <= sum(H(1:class)))
            sub_label(i) = class;
            break;
        end
    end
    start = sum(nh(1:i));
end
clear temp;
% calculate the distance between a test samples and a subclass and
% determine the class label according to the smallest distance. 
rec=0;
for i=1:Ntest
    temp = repmat(test(i,:),HH,1)-slicemean;
    temp=temp*temp';
    dist=diag(temp);
    [d, I]=sort(dist);
    I=I(1);
    found=sub_label(I);
    if (found == correct_class(i))
        rec=rec+1;    
    end
end
rate = rec/Ntest;
    
    