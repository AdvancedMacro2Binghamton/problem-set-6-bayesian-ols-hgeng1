clear all
close all
clc
%Import Data
DATA=importdata('data.csv',',',1);
data=DATA.data;
%Parameters
[N,k]=size(data);
df=N-k;
iter=10000;
r=0.12;

%1. Initial OLS
iota=ones(N,1);                       %constant
X=[iota,data(:,2:k)];                 %regressor matrix 
Y=data(:,1);                          %dependent variable
beta_ini=inv(X'*X)*X'*Y;              %beta_ols
res=Y-X*beta_ini;                     %residal
RSS=sum(res'*res);                    %residual sum of square
sigma_ini=RSS/df;                     %MSE/df=sigma^2 hat
vcv_beta_ini=inv(X'*X)*sigma_ini;     %Homoske VCV estimates
var_beta_ini=diag(vcv_beta_ini);      %variance of beta_ols
se_beta_ini=sqrt(var_beta_ini);       %standard error of beta_ols
s0=sqrt(sigma_ini);                   %sigma epsilon hat
sigmaVar=2/df*(sigma_ini^2);          %variance of sigma
                           
%2. (A) Metropolis-Hastings algorithm with flat prior
theta=[beta_ini',sigma_ini];
Sigma=[var_beta_ini;sigmaVar];
Sigma=diag(Sigma);                    %VCV of all estimators
mu = zeros(1,length(theta));
Sigma_adj=Sigma*r;

%Postier
THETA=zeros(iter,length(theta));
THETA(1,:)=theta;
accp=zeros(iter,1);
prop=zeros(iter,length(theta));
prop(1,:)=theta;

for ii=1:(iter-1)
 prop(ii+1,:)=THETA(ii,:)+mvnrnd(mu,Sigma_adj);
 while prop(ii+1,length(theta))<=0
     prop(ii+1,:)=THETA(ii,:)+mvnrnd(mu,Sigma_adj);
 end
 ratio=exp(logL(Y,X,THETA(ii,1:6)',THETA(ii,7))...
     -logL(Y,X,prop(ii+1,1:6)',prop(ii+1,7))); %Flat prior
 u=rand;
 if u<ratio
     accp(ii+1)=1;
     THETA(ii+1,:)=prop(ii+1,:);
 else accp(ii+1)=0; THETA(ii+1,:)=THETA(ii,:);
 end
end

r_acc=sum(accp)/iter;                 %accpet rate of flat prior

%2. (B) Metropolis-Hastings algorithm with given prior
%prior of beta_educ
mean=0.06;                            %mean of prior
alpha=0.05;                           %significant level
cv=norminv(1-alpha/2);                %RHS critical value
CI_h=0.085;                           %higher bound of confidence interval
sd=2*((CI_h-mean)/cv);                %se of beta_educ (2 times previous se)
p=@(x)(log(normpdf(x,mean,sd)));      %pdf of prior

%Postier
THETA1=zeros(iter,length(theta));
THETA1(1,:)=theta;
accp1=zeros(iter,1);
prop1=zeros(iter,length(theta));
prop1(1,:)=theta;

for ii=1:(iter-1)
 prop1(ii+1,:)=THETA1(ii,:)+mvnrnd(mu,Sigma_adj);
 while prop1(ii+1,length(theta))<=0
     prop1(ii+1,:)=THETA1(ii,:)+mvnrnd(mu,Sigma_adj);
 end
 %normal prior of beta_educ and flat prior of others(change)
 ratio=exp(logL(Y,X,THETA1(ii,1:6)',THETA1(ii,7))...
     -logL(Y,X,prop1(ii+1,1:6)',prop1(ii+1,7))...
     +p(prop1(ii+1,2))-p(THETA1(ii,2)));
 u=rand;
 if u<ratio
     accp1(ii+1)=1;
     THETA1(ii+1,:)=prop1(ii+1,:);
 else accp1(ii+1)=0; THETA1(ii+1,:)=THETA1(ii,:);
 end
end

r_acc1=sum(accp1)/iter;        %accpet rate

%Graphs with flat prior
figure(1)
subplot(2,4,1)
histfit(THETA(:,1),100,'kernel')
hold on
line([theta(1) theta(1)],ylim,'Color','c','LineWidth',1)
title('\beta_0')
hold off

subplot(2,4,2)
histfit(THETA(:,2),100,'kernel')
hold on
line([theta(2) theta(2)],ylim,'Color','c','LineWidth',1)
title('\beta_{educ}')
hold off

subplot(2,4,3)
histfit(THETA(:,3),100,'kernel')
hold on
line([theta(3) theta(3)],ylim,'Color','c','LineWidth',1)
title('\beta_{exp}')
hold off

subplot(2,4,4)
histfit(THETA(:,4),100,'kernel')
hold on
line([theta(4) theta(4)],ylim,'Color','c','LineWidth',1)
title('\beta_{SMSA}')
hold off

subplot(2,4,5)
histfit(THETA(:,5),100,'kernel')
hold on
line([theta(5) theta(5)],ylim,'Color','c','LineWidth',1)
title('\beta_{black}')
hold off

subplot(2,4,6)
histfit(THETA(:,6),100,'kernel')
hold on
line([theta(6) theta(6)],ylim,'Color','c','LineWidth',1)
title('\beta_{south}')
hold off
 
subplot(2,4,7)
histfit(THETA(:,7),100,'kernel')
hold on
line([theta(7) theta(7)],ylim,'Color','c','LineWidth',1)
title('\sigma_{\epsilon}^2')
hold off

%Graphs with a given prior
figure(2)
subplot(2,4,1)
histfit(THETA1(:,1),100,'kernel')
hold on
line([theta(1) theta(1)],ylim,'Color','c','LineWidth',1)
title('\beta_0')
hold off
 
subplot(2,4,2)
histfit(THETA1(:,2),100,'kernel')
hold on
line([theta(2) theta(2)],ylim,'Color','c','LineWidth',1)
title('\beta_{educ}')
hold off

subplot(2,4,3)
histfit(THETA1(:,3),100,'kernel')
hold on
line([theta(3) theta(3)],ylim,'Color','c','LineWidth',1)
title('\beta_{exp}')
hold off

subplot(2,4,4)
histfit(THETA1(:,4),100,'kernel')
hold on
line([theta(4) theta(4)],ylim,'Color','c','LineWidth',1)
title('\beta_{SMSA}')
hold off

subplot(2,4,5)
histfit(THETA1(:,5),100,'kernel')
hold on
line([theta(5) theta(5)],ylim,'Color','c','LineWidth',1)
title('\beta_{black}')
hold off

subplot(2,4,6)
histfit(THETA1(:,6),100,'kernel')
hold on
line([theta(6) theta(6)],ylim,'Color','c','LineWidth',1)
title('\beta_{south}')
hold off
 
subplot(2,4,7)
histfit(THETA1(:,7),100,'kernel')
hold on
line([theta(7) theta(7)],ylim,'Color','c','LineWidth',1)
title('\sigma_{\epsilon}^2')
hold off