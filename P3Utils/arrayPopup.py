from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import matplotlib.pyplot as plt
import numpy as np
import sys
from multiprocessing import Process
from functools import partial
import matplotlib
matplotlib.rcParams['backend.qt4']='PyQt4'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PopUp(QtWidgets.QMainWindow):
    def __init__(self, parent=None,plotFunc=None,title='', showMe=True):
        self.parent = parent
        if self.parent == None:
            self.app = QtWidgets.QApplication([])  
        super(PopUp,self).__init__(parent)
        self.setWindowTitle(title)
        self.plotFunc = plotFunc
        self.create_main_frame(title)
        self.create_status_bar()
        if plotFunc != None:
            #plotFunc(fig=self.fig,axes=self.axes)
            plotFunc(self)
        if showMe == True:
            self.show()

    def draw(self):
        self.fig.canvas.draw()

    def create_main_frame(self,title):
        self.main_frame = QtWidgets.QWidget()
      # Create the mpl Figure and FigCanvas objects. 
        self.dpi = 100
        self.fig = Figure((5, 5), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.axes = self.fig.add_subplot(111)
        #self.axes.set_title(title)

        # Create the navigation toolbar, tied to the canvas
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.canvas)
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

    def create_status_bar(self):
        self.status_text = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status_text, 1)


    def plotArray(self,image,normNSigma=3,title='',showColorBar=True,**kwargs):
        self.image = image
        if not 'vmax' in kwargs:
            goodImage = image[np.isfinite(image)]
            kwargs['vmax'] = np.mean(goodImage)+normNSigma*np.std(goodImage)
        if not 'cmap' in kwargs:
            defaultCmap=matplotlib.cm.gnuplot2
            defaultCmap.set_bad('0.15')
            kwargs['cmap'] = defaultCmap
        if not 'origin' in kwargs:
            kwargs['origin'] = 'lower'

        if 'button_press_event' in kwargs:
            cid = self.fig.canvas.mpl_connect('button_press_event',partial(kwargs.pop('button_press_event'),self))
            
        self.handleMatshow = self.axes.matshow(image,**kwargs)
        if showColorBar:
            self.fig.cbar = self.fig.colorbar(self.handleMatshow)
            cid = self.fig.canvas.mpl_connect('scroll_event', self.onscroll_cbar)
            cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick_cbar)
        self.axes.set_title(title)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)



        self.draw()

    def show(self):
        super(PopUp,self).show()
        if self.parent == None:
            self.app.exec_()

    def hoverCanvas(self,event):
        if event.inaxes is self.axes:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if row < np.shape(self.image)[0] and col < np.shape(self.image)[1]:
                self.status_text.setText('({:d},{:d}) {}'.format(col,row,self.image[row,col]))
            

    def create_status_bar(self):
        self.status_text = QtWidgets.QLabel("Awaiting orders.")
        self.statusBar().addWidget(self.status_text, 1)
        
    def onscroll_cbar(self, event):
        if event.inaxes is self.fig.cbar.ax:
            increment=0.05
            currentClim = self.fig.cbar.mappable.get_clim()
            currentRange = currentClim[1]-currentClim[0]
            if event.button == 'up':
                if QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (currentClim[0]+increment*currentRange,currentClim[1])
                elif QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (currentClim[0],currentClim[1]+increment*currentRange)
            if event.button == 'down':
                if QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (currentClim[0]-increment*currentRange,currentClim[1])
                elif QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (currentClim[0],currentClim[1]-increment*currentRange)
            self.fig.cbar.mappable.set_clim(newClim)
            self.fig.canvas.draw()

    def onclick_cbar(self,event):
        if event.inaxes is self.fig.cbar.ax:
            self.fig.currentClim = self.fig.cbar.mappable.get_clim()
            lower = self.fig.currentClim[0]
            upper = self.fig.currentClim[1]
            fraction = event.ydata
            currentRange = upper-lower
            clickedValue = lower+fraction*currentRange
            extrapolatedValue = lower+event.ydata*currentRange
            if event.button == 1:
                if QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (clickedValue,upper)
                elif QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (lower,clickedValue)
            if event.button == 3:
                if QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = ((lower-fraction*upper)/(1.-fraction),upper)
                elif QtWidgets.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (lower,lower+currentRange/fraction)
            self.fig.cbar.mappable.set_clim(newClim)
            self.fig.canvas.draw()

def plotArray(*args,**kwargs):
    #Waring: Does not play well with matplotlib state machine style plotting!
    block = kwargs.pop('block',False)
    def f(*args,**kwargs):
        form = PopUp(showMe=False)
        form.plotArray(*args,**kwargs)
        form.show()
    if block==True:
        f(*args,**kwargs)
        return None
    else:
        proc = Process(target=f,args=args,kwargs=kwargs)
        proc.start()
        return proc


def pop(*args,**kwargs):
    #Waring: Does not play well with matplotlib state machine style plotting!
    block = kwargs.pop('block',False)
    def f(*args,**kwargs):
        form = PopUp(showMe=False,*args,**kwargs)
        form.show()
    if block==True:
        f(*args,**kwargs)
        return None
    else:
        proc = Process(target=f,args=args,kwargs=kwargs)
        proc.start()
        return proc

def main():
    print('non-blocking PopUp A, close when done')
    plotArray(title='A',image=np.arange(12).reshape(4,3))
    plotArray(title='C',image=np.arange(12).reshape(3,4))
    print('blocking PopUp, close when done')
    form = PopUp(showMe=False,title='B')
    form.plotArray(np.arange(9).reshape(3,3))
    form.show()
    print('done')


if __name__ == "__main__":
    main()
