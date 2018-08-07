import os
from flask import Flask, send_file, render_template, jsonify, request, url_for, g, session, redirect
import json

app = Flask(__name__)
app.debug = True
app.config['JSON_SORT_KEYS'] = False
app.secret_key = os.urandom(24)


def saveFile(name, newlist):
    with open(name + 'J.json', 'w') as outfile:
        json.dump(newlist, outfile, indent=4, separators=(',', ': '))

def loadFile(name):
    with open(name + 'J.json', 'r') as infile:
        return json.load(infile)




@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.pop('user', None)
        userlist 
        if request.form['password'] == '':
                if request.form['username'] in userlist:
                    session['user'] = request.form['username']
        return redirect(url_for('database'))
    return render_template('index.html')

@app.route('/database')
def database():
    if g.user:
        return render_template('database.html')
    return redirect(url_for('index'))

@app.route('/targetbuttons')
def targetbuttons():
    if g.user:
        return render_template('targetbuttons.html')
    return redirect(url_for('index'))

@app.route('/datarecorder', methods = ['GET', 'POST'])
def datarecorder():
    if g.user:
        newtargets = loadFile('2018B')
        data = {}
        if request.method == 'POST':
           data['Target'] = request.form['Target']
           data['Type'] = request.form['Type']
           data['J mag'] = request.form['J mag']
           data['Filters'] = request.form['Filters']
           data['Time Windows'] = request.form['Time Windows']
           data['Nearby Laser Cals'] = request.form['Nearby Laser Cals']
           data['Important Notes'] = request.form['Important Notes']
           data['Number of Dithers'] = request.form['Number of Dithers']
           data['AO Saved State'] = request.form['AO Saved State']
           data['BIN File Range'] = request.form['BIN File Range']
           newtargets.append(data)
           saveFile('2018B', newtargets)
           print(newtargets)

           

        return render_template('datarecorder.html')

@app.route('/runbuttons', methods = ['GET', 'POST'])
def runbuttons():
    if g.user:
        if request.method == 'POST':
            filename = request.form['newFile'] + 'J.json'
            with open(filename, 'w') as json_file:
                json_file.write('[]')
        return render_template('runbuttons.html')
    return redirect(url_for('index'))

@app.route('/datebuttons')
def datebuttons():
    if g.user:
        return render_template('datebuttons.html')
    return redirect(url_for('index'))


@app.route('/calbuttons')
def calbuttons():
    if g.user:
        return render_template('calbuttons.html')
    return redirect(url_for('index'))



@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']
    #return send_file(os.path.join(app.config['/'], 'database.html'))

@app.route('/getsession')
def getsession():
    if 'user' in session:
        return session['user']

    return 'Not logged in!'


@app.route('/dropsession')
def dropsession():
    session.pop('user', None)
    return 'Dropped!'


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