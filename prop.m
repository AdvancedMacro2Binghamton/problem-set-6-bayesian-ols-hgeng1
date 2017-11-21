function f=prop(theta0,nsample,r,Sigma)
% theta0: initial guess of theta
% nsample? # of iteration or sample number
% r: adjusted scalor of VCV
% Sigma: VCV of estimators
T=zeros(nsample,length(theta0));
T(1,:)=theta0;
Sigma_adj=Sigma*r;
mu = zeros(1,length(theta0));
for jj=2:nsample
     T(jj,:)=T(jj-1,:)+mvnrnd(mu,Sigma_adj);
 while T(jj,length(theta0))<=0
      T(jj,:)=T(jj-1,:)+mvnrnd(mu,Sigma_adj);
 end
end
f=T;
end

