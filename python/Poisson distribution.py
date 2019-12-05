import matplotlib.pyplot as plt
import random
import numpy as np

# Markov chain in birth and death process(Time is discrete)

M=[	[0.2,0.8,0  ,0  ,0  ,0  ],
	[0.3,0.3,0.4,0  ,0  ,0  ],
	[0  ,0.5,0.4,0.1,0  ,0  ],
	[0  ,0  ,0.2,0.3,0.5,0  ],
	[0  ,0  ,0  ,0.4,0.4,0.2],
	[0  ,0  ,0  ,0  ,0.5,0.5] ]
'''
Markov transfer probability matrix 
Because it is a birth and death process, from time T to T+1, state i can 
only transfer to state i+1 or i-1 or keep its state.
Here are only 6 states in sample: 0,1,2,3,4,5
That is a One-dimensional Random Walk. 
'''

Pi=[]  #set a statistic frequency for the number of each states
for i in range(0,6):
	Pi.append(0)  

S=0  #set the initial state is 0

#If we expand the time T to 100000

for i in range(0,100000):  #repeat the random process 100000 times
	x=random.uniform(0,1)
	if S!=0:
		if x<=M[S][S-1]:
			S=S-1
	if S!=5:
		if x>=1-M[S][S+1]:
			S=S+1
	Pi[S]+=1

print('Pi = ',end='')
for i in range(0,6):  #Normalization
	Pi[i]=Pi[i]/100000
print(Pi)

for i in range(0,5):
	print('pi%d * p%d%d = %.4f'%(i,i,i+1,Pi[i]*M[i][i+1]),4,sep='',end='\t')
	print('pi%d * p%d%d = %.4f'%(i+1,i+1,i,Pi[i+1]*M[i+1][i]),4,sep='')
'''
We compare the value of pi[i]*p[i][i+1] and pi[i+1]*p[i+1][i]
'''