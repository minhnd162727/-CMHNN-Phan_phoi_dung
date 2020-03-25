import numpy as np
import math

# Tạo ma trận xác suất chuyển
P=np.array([0.28,0.24,0.24,0.24,22/75,0.12,22/75,22/75,16/75,16/75,0.36,16/75,0.24,0.24,0.24,0.28])
size=int(math.sqrt(len(P)))
P=P.reshape(size,size)

# Tạo A=(P^t-E)và hàng cuối cùng là 1
A=np.transpose(P)-np.eye(size)
A=np.append(A,np.ones((1,size)),axis=0)

# Tạo cột B=[0,0,...,0,1]
B=np.zeros((size,1))
B=np.append(B,[[1]],axis=0)

#Tính pp dừng pi=(A^T*A)^(-1)*(A^t*B)
AT=np.transpose(A)
AT_A=AT.dot(A)
AT_B=AT.dot(B)

pi=np.linalg.inv(AT_A).dot(AT_B)

print(np.transpose(pi))