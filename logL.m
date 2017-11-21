function f=logL(Y,X,beta,sd)
%Y: log wage from data
%beta: coeffient vector
%sd: estimated standard deviation of residuls
Y_hat=X*beta;
res_l=Y-Y_hat;
sqt=sqrt(sd);
params=[0;sqt];
f=normlike(params,res_l);
end