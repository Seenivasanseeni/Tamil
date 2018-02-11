import sys
import os
from shutil import copyfile
chars=[int(char) for char in input("Enter all characters nneded to be Created:").strip().split(" ")]
print(chars)
def is_nedded(char):
	global chars
	for char_n in chars:
		if(char==char_n):
			return True
	return False
root_s="tamil_dataset_offline"
root_d="dataset_t"
try:
	os.mkdir(root_d)
except:
	pass

for user in os.listdir(root_s):
	folder_s=root_s+"/"+user
	folder_d=root_d+"/"+user
	try:
		os.mkdir(folder_d)
	except:
		pass
	for file in os.listdir(folder_s):
		filepath_s=folder_s+"/"+file
		filepath_d=folder_d+"/"+file
		try:
			if(is_nedded(int(file[:3]))):
				print(filepath_d)
				copyfile(filepath_s,filepath_d)
		except:
			pass
