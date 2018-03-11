import configparser

config=configparser.ConfigParser()
config.read("param.ini")

def get(param):
	try:
		return config["dataset"][param]
	except :
		print("Section/key not found check param.ini file for errors")
