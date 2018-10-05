import os
from flask import Flask, send_file, render_template, jsonify, request, url_for, g, session, redirect
from glob import glob
import json
from pathlib import Path
from datetime import datetime


app = Flask(__name__)
app.debug = True
app.config['JSON_SORT_KEYS'] = False
app.secret_key = os.urandom(24)


def saveFile(name, newlist):
    with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/' + name + '.json', 'w') as outfile:
        json.dump(newlist, outfile, indent=4, separators=(',', ': '))

def saveCalFile(name, newlist):
    with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/calfiles/' + name + '.json', 'w') as outfile:
        json.dump(newlist, outfile, indent=4, separators=(',', ': '))

def loadFile(name):
    with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/' + name + '.json', 'r') as infile:
        return json.load(infile)

def loadCalFile(name):
    with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/calfiles/' + name + '.json', 'r') as infile:
        return json.load(infile)



@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.pop('user', None)
        userlist = ['clarissardoo']
        if request.form['password'] == 'mkidsecure':
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
        data = []
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/*.json')
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                    data1 = json.load(datarunner)
            for elem in data1:
                    data.append(elem['Target'])
        return render_template("targetbuttons.html", data = set(data))
    return redirect(url_for('index'))

@app.route('/datarecorder', methods = ['GET', 'POST'])
def datarecorder():
    if g.user:
        data = {}
        if request.method == 'POST':
           newtargets = loadFile(request.form['FileName'])
           data['Target'] = request.form['Target']
           data['Type'] = request.form['Type']
           data['J mag'] = request.form['J mag']
           data['Sunset Date(s)'] = request.form['Sunset Date(s)']
           data['Seeing (AO)'] = request.form['Seeing (AO)']
           data['Filters'] = request.form['Filters']
           data['Time Windows'] = request.form['Time Windows']
           data['Nearby Laser Cals'] = request.form['Nearby Laser Cals']
           data['Important Notes'] = request.form['Important Notes']
           data['Number of Dithers'] = request.form['Number of Dithers']
           data['AO Saved State'] = request.form['AO Saved State']
           data['BIN File Range'] = request.form['BIN File Range']
           newtargets.append(data)
           saveFile(request.form['FileName'], newtargets)
           #print(newtargets)
        return render_template('datarecorder.html')


@app.route('/calrecorder', methods = ['GET', 'POST'])
def calrecorder():
    if g.user:
        data = {}
        if request.method == 'POST':
            newCals = loadCalFile(request.form['File_Name'])
            data['Target'] = request.form['calTarget']
            data['Run'] = request.form['Run']
            data['Time'] = request.form['Time']
            data['808nm'] = request.form['808nm']
            data['904 nm'] = request.form['904 nm']
            data['920 nm'] = request.form['920 nm']
            data['980 nm'] = request.form['980 nm']
            data['1120 nm'] = request.form['1120 nm']
            data['Rethresholding range'] = request.form['Rethresholding range']
            newCals.append(data)
            saveCalFile(request.form['File_Name'], newCals)
            print(newCals)
        return render_template('calrecorder.html')

@app.route('/runbuttons', methods = ['GET', 'POST'])
def runbuttons():
    if g.user:
        fileList = []
        if request.method == 'POST':
            filename = request.form['newFile'] + '.json'
            print(filename)
            with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/' + filename, 'w') as json_file:
                json_file.write('[]')

        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/*.json')
        for elem in jsonfiles:
            _, tail = os.path.split(elem)
            fileList.append(tail[:-5    ])
        fileList.remove('2016B')
        fileList.remove('2017A')
        fileList.remove('2017B')
        fileList.remove('2018A')
        return render_template('runbuttons.html', data = fileList)
    return redirect(url_for('index'))


@app.route('/datebuttons')
def datebuttons():
    if g.user:

        data = []
        prettyData = []
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/*.json')
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                    data1 = json.load(datarunner)
            for elem in data1:
                    dates = elem['Sunset Date(s)']
                    datelist = dates.split("<br />")
                    data.extend(datelist)
                    print(elem['Sunset Date(s)'])

            for elem in data:
                datetimeobject = datetime.strptime(elem,'%Y%m%d')
                newformat = datetimeobject.strftime('%B-%d-%Y')
                prettyData.append(newformat)


        return render_template("datebuttons.html", data = set(data))


        #return render_template('datebuttons.html')
    return redirect(url_for('index'))


@app.route('/calbuttons', methods = ['GET', 'POST'])
def calbuttons():
    if g.user:
        data = []
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/calfiles/*.json')
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                data1 = json.load(datarunner)
            for elem in data1:
                data.append(elem['Target'])

        if request.method == 'POST':
            file_name = request.form['newCalFile'] + '.json'
            print(file_name)
            with open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/calfiles/' + file_name, 'w') as jsonfile:
                jsonfile.write('[]')

        return render_template("calbuttons.html", data = set(data))

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
        data = []
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/*.json')
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                    data1 = json.load(datarunner)
            for elem in data1:
                print(request.form['submit'])
                print(request.form)
                if elem.get("Target") == request.form['submit']:
                    data.append(elem)

        return render_template("targets.html", data = data)


@app.route('/runfinder', methods=['POST'])
def runfinder():
    if request.method == 'POST':
        datarunner = open('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/' + request.form['submit']+'.json')
        data1 = json.load(datarunner)
        return render_template("runs.html", data = data1)

@app.route('/calibrationfinder', methods=['POST'])
def calibrationfinder():
    if request.method == 'POST':
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/calfiles/*.json')
        data = []
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                    data1 = json.load(datarunner)
            for elem in data1:
                if elem.get("Target") == request.form['submit']:
                    data.append(elem)   
        return render_template("cals.html", data = data)



@app.route('/datefinder', methods=['POST'])
def datefinder():
    if request.method == 'POST':
        jsonfiles = glob('/Users/clarissardoo/Desktop/Clarity/MKIDPipeline/mkidpipeline/clarity/datafiles/*.json')
        data = []
        for recordYear in jsonfiles:
            with open(recordYear) as datarunner:
                    data1 = json.load(datarunner)
            for elem in data1:
                date = request.form['submit']
                if date in elem.get("Sunset Date(s)"):
                    data.append(elem)   
        return render_template("dates.html", data = data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 3000)