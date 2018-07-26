import json
from pprint import pprint

run = input("What run would you like information on?\n(16B, 17A, 17B, 18A, ALL)\n")

if run == '17A':
	name = input("What target would you like information?\n--STAR LIST:--\nSAO82875\nHD77407\nSAO100725\nTau Boo\nCandidate 1\nHD114174\nSAO65921\nSAO65890\nHD91782\nHD148112\nSAO83893\nType here: ")

elif run == '18A':
	name = input("What target would you like information?\n--STAR LIST:--\nSAO65921\nSAO44531\nTau Boo\nVega\nOmega Her\nSAO47913\nHD114174\nWhite Light\nType here: ")

elif run == '17B':
	name = input("What target would you like information on?\n--STAR LIST:--\nSAO90440 (32 Peg)\nSAO90717\nHR8799\nSAO108402\nSAO54058 (Delta And)\nSAO112345 (HD32297)\nSAO041363\nSAO 37734 (Gamma And)\n Type here: ")

elif run == '16B':
	name = input("What target would you like information on?\n--STAR LIST:--\nSAO61424\nHR8799\nEps Eridani\nSAO78135 (Propus)\nSAO42642\nType here: ")

elif run == 'ALL':
	name = input("What target would you like information on?\n--STAR LIST:--\nSAO61424\nHR8799\nEps Eridani\nSAO78135 (Propus)\nSAO42642\nSAO82875\nHD77407\nSAO100725\nTau Boo\nCandidate 1\nHD114174\nSAO65921\nSAO65890\nHD91782\nHD148112\nSAO83893\nSAO90440 (32 Peg)\nSAO90717\nSAO108402\nSAO54058 (Delta And)\nSAO112345 (HD32297)\nSAO041363\nSAO 37734 (Gamma And)\nSAO44531\nVega\nOmega Her\nSAO47913\nWhiteLight\nType star here: ")

if run == 'ALL':
	filelist = ['2016B', '2017A', '2017B', '2018A']
else:
	filelist = ['20'+run]


def makeList(stri):
	out = []
	buff = []
	for c in stri:
	    if c == '\n':
	        out.append(''.join(buff))
	        buff = []
	    else:
	        buff.append(c)
	else:
	    if buff:
	       out.append(''.join(buff))

	return out


def prettyPrintList(list, year):
	#print(list)
	i = 1
	for subLists in list:
		print("\n----------\nDATE #" + str(i), "for", name, "on", year, "run:",
			"\n----------\n")
		i+=1
		for e in subLists:
			print(e)


for recordYear in filelist:
	#print("reading")
	with open(recordYear+'J.json') as datarunner:
		data1 = json.load(datarunner)
		bigList = []
		for entry in data1:
			#

			if entry.get("Target") == name:
				#print("BINGO")
				bigList = [[] for _ in range(len(makeList(entry.get('Filters'))))]
				#print(bigList)
				for elem in entry:
						myStr = elem + ": \n\t"
						myList = makeList(entry.get(elem))

						for i in range(0, len(myList)):
							bigList[i].append(myStr+myList[i])

				break
		prettyPrintList(bigList,recordYear)
		


