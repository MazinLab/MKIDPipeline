import os
from flask import Flask, send_file, render_template, jsonify, request
import json

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


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def homepage():
	return render_template('database.html')
	#return send_file(os.path.join(app.config['/'], 'database.html'))


@app.route('/targetfinder', methods=['POST'])
def targetfinder():
	if request.method == 'POST':
		filelist = ('2016B', '2017A', '2017B', '2018A')
		data = []
		for recordYear in filelist:
			with open(recordYear+'J.json') as datarunner:
					data1 = json.load(datarunner)
			for elem in data1:
				if elem.get("Target") == request.form['submit']:
					data.append(elem)
		return render_template("targets.html", data = data)


@app.route('/runfinder', methods=['POST'])
def runfinder():
	if request.method == 'POST':
		datarunner = open(request.form['submit']+'J.json')
		data1 = json.load(datarunner)
		return render_template("runs.html", data = data1)

@app.route('/calibrationfinder', methods=['POST'])
def calibrationfinder():
	if request.method == 'POST':
		filelist = ('Wavecals_2017A', 'WaveCals_2017B', 'WaveCals_2018A')
		data = []
		for recordYear in filelist:
			with open(recordYear+'.json') as datarunner:
					data1 = json.load(datarunner)
			for elem in data1:
				if elem.get("Target") == request.form['submit']:
					data.append(elem)	
		return render_template("cals.html", data = data)


@app.route('/datefinder', methods=['POST'])
def datefinder():
	if request.method == 'POST':
		filelist = ('2016B', '2017A', '2017B', '2018A')
		data = []
		for recordYear in filelist:
			with open(recordYear+'J.json') as datarunner:
					data1 = json.load(datarunner)
			for elem in data1:
				date = request.form['submit']
				if date in elem.get("Sunset Date(s)"):
					data.append(elem)	
		return render_template("dates.html", data = data)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 3000)