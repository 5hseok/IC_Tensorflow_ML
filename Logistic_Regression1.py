import numpy as np
x=np.array([3,3,1,4,4])         #input(학생의 등급)
print("rank = ",x)
y=np.array([0,1,1,1,0])         #result(입학 허가 여부)
print("admit = ",y)
ones=np.ones(y.shape)           #cost와 lost함수에서 1-연산을 위해 만듬
xm=x*np.array([-1,-1,-1,-1,-1]) #지수 함수에 -제곱 연산을 위해 x에 -1을 곱함 

theta = 0.001                   #초기 세타값을 0.001로 둠
learningrate = 0.1              #초기 w 값을 0.1로 둠

for i in range(10):             #epoch(시행횟수)는 10번
    yest=1.0/(1+np.exp(theta*xm))                       #==sigmoid(np.transpose(theta)*x) == y hat
    cost = y*np.log(yest)-(ones-y)*np.log(ones-yest)    #y가 0이라면 뒤의 항, 1이라면 앞의 항을 만나 cost에 들어감
    ll=-sum(cost)/5                                     # Lost function이다. -1/m * sum
    #print("log likelyhood = ",ll)
    dtheta = sum((yest-y)*x)/5
    theta=theta-learningrate*dtheta
    print("new theta = ",theta)
    