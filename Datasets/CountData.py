import sys
import os
import pickle
root_directory = sys.argv[1]
print("Script Running fot:",root_directory)

letter_count={}
usr_count=0
corrupt_count=0
total=0
for usr in os.listdir(root_directory):
	usr_count+=1;
	for img in os.listdir(root_directory+"/"+usr):
		total+=1
		if(not(len(img)==11)):
			corrupt_count+=1
			continue
		label=int(img[:3])
		try:
			letter_count[label]+=1
		except:
			letter_count[label]=0


print("Total",total)
print("No of users:",usr_count)
print("Corrupt:",corrupt_count)
print("Usable:",total-corrupt_count)

#visualization purpose
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(letter_count.keys(),letter_count.values())
plt.savefig(sys.argv[1]+"-Dist")
#plt.show()
