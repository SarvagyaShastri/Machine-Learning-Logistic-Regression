import sys
import random
from random import randint
import math


def read_for_splitting_data(file_name):
	temp_data=[]
	try:
		with open(file_name) as docs:
			for line in docs:
				line=line.split()
				temp_data.append(line)

	except Exception,e:
		raise e
		print "File Not Found, program will exit"
		exit()

	return temp_data

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
	#data=populate_data(temp_data,labels,max_features)
	#print len(data[0])
	return max_features



def read_train_data(i,k,file_name,max_features):
	labels=[]
	temp_data=[]
	for j in range(k):
		if j==i:
			continue
		else:
			#labels=[]
			#		temp_data=[]
			try:
				with open(file_name % j) as docs:
					for line in docs:
						line=line.split()
						temp_data.append(line[1:])
						labels.append(int(line[0]))
			except Exception:
				print "File Not Found, program will exit"
				exit()
	#max_value=[int(item[-1].split(":")[0]) for item in temp_data]
	#max_features=max(max_value)+1
	data=populate_data(temp_data,labels,max_features)
	return data,labels

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



def read_test_data(i,file_name, max_features):
	labels=[]
	temp_data=[]
	
	with open(file_name % i) as docs:
		for line in docs:
			line=line.split()
			temp_data.append(line[1:])
			labels.append(int(line[0]))
		
	#max_value=[int(item[-1].split(":")[0]) for item in temp_data]
	#max_features=max(max_value)+1
	data=populate_data(temp_data,labels,max_features)
	return data,labels

def populate_weights(max_features):
	weights=[randint(-1,1) for i in range(max_features)]
	return weights


def perceptron(data,labels,weights,epoch,rate,sigma2):
	j=0
	x=0
	while j<epoch:
		for i in range(len(data)):
			t=i+1
			actual_label=labels[i]
			dot_product=reduce(lambda x,y:x+y,map(lambda x,y:x*y,weights,data[i])) # w^{T}*x
			adapted_rate=float(rate)/(1+((rate*t)/sigma2))
			update=map(lambda x:x*actual_label,data[i]) #y_{i}*x_{i}
			coeff=-1/float(1+math.exp(actual_label*dot_product)) #-1/1+e^yi*wtx
			gradient=map(lambda x,y:x+y,map(lambda x:x*coeff,update),map(lambda x:x*float(2/sigma2),weights)) #coeff*update+2w/sigma
			update2=map(lambda x:x*adapted_rate, gradient)
			weights=map(lambda x,y:x-y,weights,update2)
			
		j+=1
	return weights

def calculate_accuracy(weights,data,labels,bias,hits,miss):
	for i in range(len(data)):
		dot_product=reduce(lambda x,y:x+y,map(lambda x,y:x*y,weights,data[i]))
		#derived_label=reduce(lambda x,y:x+y,dot_product)+bias
		actual_label=labels[i]
		#print dot_product

		if -dot_product<=0:
			predicted=1
		else:
			predicted=-1
		if predicted==actual_label:
			hits+=1
		else:
			miss+=1
	return float(hits)/(hits+miss)



def split(k):
	file_name="data/a5a.train"
	data=read_for_splitting_data(file_name)
	split_parts=len(data)/k

	for i in range(len(data)):
		if i%split_parts==0:
			if i>0:
				target.close()
			number=i/split_parts
			target=open("data/train_%d.data"%number,"w")
		target.write(' '.join(data[i]))
		target.write("\n")	



def cross_validate(k,epoch_list, c, learning_rate):
	print "Accuracy & Rate & sigma^2 & epoch"
	max_features=read_data("data/a5a.train")
	max_features+=1
	best_result=[]
	for rate in learning_rate:
		for C in c:
			cross_validation=[]
			for i in xrange(k):
				accuracies_final=[]
				file_name="data/train_%d.data"
				data,labels=read_train_data(i,k,file_name, max_features)
				new_data=[]
				for l,item in enumerate(data):
					new_data.append([1]+data[l])
				
				data=new_data
				test_data,test_labels=read_test_data(i,file_name,max_features)
				test_data=[[1]+test_data[i] for i,item in enumerate(test_data)]
				weight_initialize=populate_weights(max_features)
				initialized_bias=randint(-1,1)
				#print initialized_bias, " This is randomly initialized bias"
				for epoch in epoch_list:
					total_data=[]
					for i in range(len(data)):
						total_data.append(data[i]+[labels[i]])
					random.shuffle(total_data)
					labels=[]
					labels=[i[-1] for i in total_data]
					data=[i[:-1] for i in total_data]
					
					bias=initialized_bias
					weights=[bias]+weight_initialize
					#print len(test_data[0]), len(data[0]), "******"
					final_weight_vector=perceptron(data,labels,weights,epoch,rate,C)
					if len(test_data[0])>len(data[0]):
						difference=len(test_data[0])-len(data[0])
						final_weight_vector=final_weight_vector+[0]*difference
						#if len(test_data[0])<len(data[0]):
						#	difference=len(data[0])-len(test_data[0])
						#	final_weight_vector=final_weight_vector+[0]*difference

					#print len(final_weight_vector), len(test_data[0])
					#61.25 & 0.01 & 0.5 & 5\\ \hline
					accuracy=calculate_accuracy(final_weight_vector,test_data,test_labels,bias,0,0)
					accuracies_final.append([accuracy,rate,C, epoch])
					#print accuracy*100,"&",rate,"&",C,"&", epoch, "\\ \hline"
					weights=[]
		
				result=max(accuracies_final)
				cross_validation.append(result)
			best_acc=max(cross_validation)
			print best_acc[0]*100,"&",best_acc[1],"&",best_acc[2],"&", best_acc[3], "\\ \hline"
			best_result.append(best_acc)
			print "\n"
				#print result[0],"max accuracy for learning rate",result[1],"sigma^2",result[2]

	print max(best_result)
















				