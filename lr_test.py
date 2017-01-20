import sys
import random
from random import randint
import math
#import matplotlib.pyplot as plt
import cross_validation_splitter
def read_data(fname):
	labels=[]
	temp_data=[]
	try:
		with open(fname) as docs:
			for line in docs:
				line=line.split()
				temp_data.append(line[1:])
				labels.append(int(line[0]))
	except Exception:
		print "File Not Found, program will exit"
		exit()
	max_value=[int(item[-1].split(":")[0]) for item in temp_data]
	max_features=max(max_value)+1
	data=populate_data(temp_data,labels,max_features)
	#print len(data[0])
	return data,labels,max_features

def populate_data(data,labels,max_features):
	final_result=[]
	for row in data:
		temp_data=[0 for i in range(max_features)]
		for item in row:
			item=item.split(":")
			index=int(item[0])
			value=int(item[1])
			temp_data[index]=value
		final_result.append(temp_data)
	return final_result

def populate_weights(max_features):
	weights=[randint(-1,1) for i in range(max_features)]
	return weights

def perceptron(data,labels,weights,epoch,rate,sigma2):
	j=0
	x=0
	log_likelyhood=[]
	while j<epoch:
		log_temp=[]
		for i in range(len(data)):
			t=i+1
			actual_label=labels[i]

			dot_product=reduce(lambda x,y:x+y,map(lambda x,y:x*y,weights,data[i])) # w^{T}*x
			adapted_rate=float(rate)/(1+((rate*t)/sigma2))
			#print adapted_rate
			update=map(lambda x:x*actual_label,data[i]) #y_{i}*x_{i}
			coeff=-1/float(1+math.exp(actual_label*dot_product)) #-1/1+e^yi*wtx
			gradient=map(lambda x,y:x+y,map(lambda x:x*coeff,update),map(lambda x:x*float(2/sigma2),weights)) #coeff*update+2w/sigma
			update2=map(lambda x:x*adapted_rate, gradient)
			weights=map(lambda x,y:x-y,weights,update2)
		
			neg_log_likelyhood=math.log(1+math.exp(-1*actual_label*dot_product))
			log_temp.append(neg_log_likelyhood)
		log_likelyhood.append(reduce(lambda x,y:x+y,log_temp)+reduce(lambda x,y:x+y,map(lambda x,y:x*y,weights,map(lambda x:x*float(2/sigma2),weights))))
			
		j+=1
	return weights, log_likelyhood

def calculate_accuracy(weights,data,labels,bias,hits,miss):
	for i in range(len(data)):
		dot_product=reduce(lambda x,y:x+y,map(lambda x,y:x*y,weights,data[i]))
		actual_label=labels[i]

		if -dot_product<=0:
			predicted=1
		else:
			predicted=-1
		if predicted==actual_label:
			hits+=1
		else:
			miss+=1
	return float(hits)/(hits+miss)

def main():
	#print "hi"
	k=5
	print "Starting Cross Validation"
	cross_validation_splitter.split(k)
	epoch_list=[3,10]
	C=[50, 40, 30, 20, 10]
	learning_rate=[0.1, 0.50]
	cross_validation_splitter.cross_validate(k,epoch_list, C, learning_rate)
	C=30
	learning_rate=[0.5]

	accuracies_final=[]
	file_name="data/a5a.train"
	try:
		margin=sys.argv[1]
		assert float(margin)<float(5) and float(margin)>=0,"Value between 0.0 to 5.0 required"
	except Exception:
		margin=0
	epoch_list=[10]
	data,labels,max_features=read_data(file_name)
	new_data=[]
	for i,item in enumerate(data):
		new_data.append([1]+data[i])
		
	data=new_data
	file_name="data/a5a.test"

	test_data,test_labels,max_test_feature=read_data(file_name)
	test_data=[[1]+test_data[i] for i,item in enumerate(test_data)]
	weight_initialize=populate_weights(max_features)
	initialized_bias=randint(-1,1)
	log_likelyhood=[]
	for epoch in epoch_list:
		total_data=[]
		for i in range(len(data)):
			total_data.append(data[i]+[labels[i]])
		random.shuffle(total_data)
		labels=[]
		labels=[i[-1] for i in total_data]
		data=[i[:-1] for i in total_data]
		for rate in learning_rate:
			bias=initialized_bias
			weights=[bias]+weight_initialize
			log=[]
			final_weight_vector, log=perceptron(data,labels,weights,epoch,rate,C)
			if len(test_data[0])>len(data[0]):
				difference=len(test_data[0])-len(data[0])
				final_weight_vector=final_weight_vector+[0]*difference
	
			accuracy=calculate_accuracy(final_weight_vector,test_data,test_labels,bias,0,0)
			accuracies_final.append([accuracy,rate,epoch])
			print "Accuracy found",accuracy*100,"for rate",rate,"with epoch ",epoch," and bias of the final vector",bias
			weights=[]
	result=max(accuracies_final)
	print "\n"
	print result[0],"max accuracy for learning rate",result[1],"and epoch",result[2]
	#plt.plot(log)
	#plt.ylabel("negative log likelyhood")
	#plt.xlabel("epochs")
	#plt.show()

main()






"""
******************3_1_1*****************************
1.File read data+label
2.Append each input -->bias+example
3.weight --> [1,0,0,..]
4.replace rate --->learning rate( given in assignment)
          t ->t_th example
****************************3_1_2******************
cross  validation
learning rate ,gamma --> discussions
C ---> discussions
"""
