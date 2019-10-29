import random
import math
import matplotlib.pyplot as plt
import operator

# create a client class to save all information
class client:
	def __init__(self,arr=0,In=0,ser=0,Qsize=0,start=0,end=0,delay=0):
		self.arr=arr 		# arrival time
		self.In=In 			# arrival interval
		self.ser=ser 		# serve time
		self.Qsize=Qsize 	# the number of clients in queue
		self.start=start 	# serve start time
		self.end=end 		# serve finish time
		self.delay=delay 	# time of staying in system

# time class: in order to sort all time point
class time:
	def __init__(self,t,s):
		self.t=t 	# time
		self.s=s 	# state in this time :arrival(1), leave(-1)
# show queue process plot
def show_plot(C):
	T=[]
	for i in range(len(C)):
		T.append(time(t=C[i].arr,s=1))
		T.append(time(t=C[i].end,s=-1))
	#sort all time points
	cmpfun = operator.attrgetter('t')
	T.sort(key=cmpfun)

	# draw the plot for queue process
	X=[]
	Y=[]
	q=0
	X.append(0)
	Y.append(0)
	for i in range(len(T)):
		if T[i].s==1:
			X.append(T[i].t)
			Y.append(q)
			X.append(T[i].t)
			Y.append(q+1)
			q+=1
		else:
			X.append(T[i].t)
			Y.append(q)
			X.append(T[i].t)
			Y.append(q-1)
			q-=1
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(X, Y, c='#0099ff')
	plt.show()



def exp_dist(lamda):
# generate a time interval according to exponential distribution with a parameter lamda
# use the method of inverse transform sampling to generate time interval 
	y=random.uniform(0,lamda)
	x=math.log(y/lamda)/(-lamda)
	return x


N = 10  	#the number of client
lamda = 0.2		# parameter of exp distribution to generate time interval
				# the mean of exp distribution is 1/lamda,which is 5 in this case
C = []		#clinet list


# generate the arrival interval and serve time of all clients
# both the parameters of arrival interval and serve time are lamda. It is a stationary system 
for i in range(N):
	a=client(In=exp_dist(lamda),ser=exp_dist(lamda))
	C.append(a)


# simulate a queue process
now=0	#current time
last=0	# last finish time

# simulate the queue process
for i in range(N):
	C[i].arr=now+C[i].In
	now=C[i].arr
	if last<now:
		last=now
	C[i].start=last
	C[i].end=last+C[i].ser
	C[i].delay=C[i].end-C[i].arr
	last+=C[i].ser

# calculate the number of clients in queue
for i in range(N):
	q=0
	j=i
	while C[j].start>C[i].arr and j>=0:
		q+=1
		j-=1
	C[i].Qsize=q

# show all client information in queue process
print("No.\tInter\tserve\tarrival\tstart\tend  \tdelay\tQsize")
for i in range(N):
	print("%2d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%2d"%
		(i+1,C[i].In,C[i].ser,C[i].arr,C[i].start,C[i].end,C[i].delay,C[i].Qsize))

# show queue process
show_plot(C)

'''




fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, Y, s=10, c='#0099ff')
plt.show()

'''