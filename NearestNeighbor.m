function [classes,rec,rate] = NearestNeighbor(train, test, correct_class, category, nc,trainLabel)
% function: classify the testing data according to nearest neighbor rule
%
%input:
%train: training data
%test:  testing data
%correct_class: the true class label of the testing data
%category:  the number of classes 
%nc: category-by-1 matrix, indicationg the number of samples in each class
%    for training data
%
%output:
%rate: recognition rate

n_test = size(test,1);
n_train = size(train,1);
classes=zeros(n_test,size(category,1));
rec = 0;
for i = 1:n_test 
    
    temp = repmat(test(i,:),n_train,1)-train;
    temp = temp.^2;
    dist = sum(temp',1);
                
    %[val,I]=min(dist);
    [val,I] = sort(dist);
    %val = val(1);
    %I = I(1);
    found=0;
    
    for count = 1:(size(train,2)/4)
        Icount = I(count);
        classSum=0;
        for class = 1:category
            classSum = classSum+nc(class);
            if(classSum>=Icount)
                found=trainLabel(Icount);
                if (found == correct_class(i))
                    rec=rec+1;
                    classes(i)=found;
                    
                end
                break;
            end
        end
        if (found == correct_class(i))
             break;
        end
    end            
   
               
    
end
rate = rec/n_test;