accp=zeros(iter+1,1);
accp(1)=1;

for ii=1:iter
 value(ii,:)=theta;
 prop=prop(value(ii,:));
 ratio=exp(logL(prop(1:6),prop(7))-logL(theta(1:6),theta(7));
 u=rand;
 if u<ratio
     accp(ii+1)=1;
     theta=prop;
 else accp(ii+1)=0; theta=theta;
 end
end

accp=accp(1:iter-1,:);
r_acc=sum(accp)/iter;        %accpet rate
post=zeros(sum(accp),length(theta))
for jj=1:length(theta)
    A=value(:,jj);
    post(:,jj)=A(accp==1);
end
