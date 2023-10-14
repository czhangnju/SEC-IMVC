function [G, Y] = SEC_IMVC(F, index, param)

alpha = param.alpha; 
beta = param.beta; 
lambda = param.lambda; 
r  = 2;

c = param.cls_num;
[n, num_view] = size(index);

W = cell(1, num_view); G = W; E=W; Z=W;
for iv = 1:num_view
    pos0{iv} = find(index(:,iv) == 0);
    pos1{iv} = find(index(:,iv) == 1);  
    I = eye(n);
    I(pos0{iv},:) = [];
    W{iv} = I;
    
    I = eye(n);
    I(:,pos0{iv}) = [];
    O{iv} = I;    
    E{iv} = zeros(size(F{iv}));    
    G{iv} = O{iv}*F{iv};
    R{iv} = eye(c);
end
clear I iv 

sY = 0;
for iv = 1:num_view
   sY = sY + G{iv};
end
mm = max(sY,[],2);
Y = 1 - double(sY < mm(:));
clear sY 

w = ones(1, num_view)/num_view;

MAXITER = 15;

opts.info = 1;
opts.gtol = 1e-5;
for iter = 1:MAXITER
%     if mod(iter, 10)==0
%     fprintf('%d... ',iter);
%     end
    
     % Gv step
    GG = 0;
    for iv = 1:num_view
        GG = GG + G{iv}*G{iv}';
    end
    for iv = 1:num_view
        tempG = -lambda*(GG-G{iv}*G{iv}');
        tempG2 = -beta*w(iv)^r*R{iv}*Y' - (F{iv}-E{iv})'*W{iv};
        tempG2 = tempG2';
        G{iv} = FOForth(G{iv}, tempG2, @fun,opts, tempG, tempG2);
    end
    clear tempG tempG2
    
       % Ev step
    for iv=1:num_view
        E{iv} = prox_l12(F{iv} - W{iv}*G{iv}, alpha/2);
    end
    

    
    % R step
    for iv = 1:num_view
        [Ut,~, St]=svd(G{iv}'*Y,'econ');
        R{iv} = Ut*St';      
    end
    clear tempU Ut St
    
         % Y step
    sY = 0;
    for iv = 1:num_view
        sY = sY + w(iv)^r*G{iv}*R{iv};
    end
    mm = max(sY,[],2);
    Y = 1 - double(sY < mm(:));
    clear sY
 

    % w step
    err = zeros(1, num_view);
    sw= 0;
    for iv = 1:num_view
        dd = Y-G{iv}*R{iv};
        err(iv) = 1/sum(sum(dd.*dd));
        sw = sw + err(iv)^(1/(r-1));
    end
    w = err.^(1/(r-1))/sw;
    clear sw err dd
  
end


end

function [funX, F] = fun(X, A, Kp)
        F = 2 * A * X + Kp;
        funX = sum(sum(X .* (A * X + Kp)));
    end