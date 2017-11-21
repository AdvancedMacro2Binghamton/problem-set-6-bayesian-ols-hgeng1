clear all
close all
clc
%Import Data
DATA=importdata('data.csv',',',1);
data=DATA.data;
%Parameters
[N,k]=size(data);
df=N-k;
iter=3000;

%Initial OLS
iota=ones(N,1); %constant
X=[iota,data(:,2:k)]; %regressor matrix 
Y=data(:,1);                              %dependent variable
beta_ini=inv(X'*X)*X'*Y;              %beta_ols
res=Y-X*beta_ini;                     %residal
RSS=sum(res'*res);                    %residual sum of square
sigma_ini=RSS/df;                     %MSE/df=sigma^2 hat
vcv_beta_ini=inv(X'*X)*sigma_ini;     %Homoske VCV estimates
var_beta_ini=diag(vcv_beta_ini);      %variance of beta_ols
se_beta_ini=sqrt(var_beta_ini);       %standard error of beta_ols
s0=sqrt(sigma_ini);                   %sigma epsilon hat
sigmaVar=2/(N-k)*(sigma_ini^2);       %variance of sigma
                           
%Metropolis-Hastings algorithm
theta=[beta_ini',sigma_ini];

%Proposals
Sigma=[var_beta_ini;sigmaVar];
Sigma=diag(Sigma);                    %VCV of all estimators
value=prop(theta,iter,2000,Sigma);
%Postier
post=zeros(iter,length(theta));
accp=zeros(iter,1);

%Flat prior
for ii=1:iter
 post(ii,:)=theta;
 prop=value(ii,:);
 ratio=exp(logL(Y,X,theta(1:6)',theta(7))-logL(Y,X,prop(1:6)',prop(7)));
 u=rand;
 if u<ratio
     accp(ii+1)=1;
     theta=prop;
 else accp(ii+1)=0; theta=theta;
 end
end

r_acc=sum(accp)/iter;        %accpet rate

figure(1)
suptitle('Posterior Distribution w/ flat prior for all')
subplot(2,4,1)
hist(post(:,1))
title('\beta_0')
 
subplot(2,4,2)
hist(post(:,2))
title('\beta_{educ}')

subplot(2,4,3)
hist(post(:,3))
title('\beta_{exp}')

subplot(2,4,4)
hist(post(:,4))
title('\beta_{SMSA}')

subplot(2,4,5)
hist(post(:,5))
title('\beta_{black}')

subplot(2,4,6)
hist(post(:,6))
title('\beta_{south}')
 
subplot(2,4,7)
hist(post(:,7))
title('\sigma_{\epsilon}')
