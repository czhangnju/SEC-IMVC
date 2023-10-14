clear; close all;

addpath ./ClusteringMeasure/
addpath ./functions
path = './data/';


Database = 'Digit';  
percentDel = 0.1;

Datafold = [path, Database];
load(Datafold)
Indexfold = [path,'Index/Index_',Database,'_percentDel_',num2str(percentDel),'.mat'];
load(Indexfold)

cls_num = numel(unique(Y));
param.cls_num = cls_num;
perf = []; gt = double(Y);
Xc = X;
ind = Index{1}; F = cell(1, length(Xc));

for i=1:length(Xc)
        Xci = Xc{i};
        Xci = NormalizeFea(Xci,0);
        indi = ind(:,i);
        pos = find(indi==0);
        Xci(:,pos)=[]; 
        Xc{i} = Xci;
        S = constructW_PKN(Xc{i}, 10, 1);
        F{i} = SpecEmbedding((abs(S)+abs(S))/2, cls_num);
end   
clear Xci i indi pos S


param.alpha = 1e-4;
param.beta = 1e-5;
param.lambda  = 5e-2;

[G, FF] = SEC_IMVC(F, ind, param);
[~, Clus] = max(FF,[],2);

[ACC,NMI,PUR] = ClusteringMeasure(gt,Clus); %ACC NMI Purity
[Fscore,Precision,R] = compute_f(gt,Clus);
[AR,~,~,~]=RandIndex(gt,Clus);
result = [ACC NMI AR Fscore PUR Precision R];
fprintf("ACC,NMI, ARI: %.4f, %.4f, %.4f \n", result(1),result(2),result(3));


