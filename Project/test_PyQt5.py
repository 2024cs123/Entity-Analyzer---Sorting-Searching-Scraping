import sys #system specific parameters and functions
"""This modules provides access to some variables used or maintained by interpreter
and to functions that interacts strongly with an interpreter it is always available"""
from PyQt5.QtWidgets import QApplication,QMainWindow,QLineEdit,QPushButton,QButtonGroup,QRadioButton,QLabel,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout #the last three deals with layout managers they are not widgets
from PyQt5.QtGui import QIcon,QFont,QPixmap
from PyQt5.QtCore import Qt

"""normally we cant add a layout manager to the mainwindow object because 
widgets have a specific design and layout structure thats normally incompatible
with layout manangers so:::
    we will create a generic widget then add layouts to that widget then add 
    that widget to the mainwindow in order to display the layout"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My cool first GUI")
        self.setGeometry(700,300,500,500)
        self.line_edit=QLineEdit(self)
        self.button=QPushButton("Submit",self)
        self.setWindowIcon(QIcon("pic.jpg"))
        label = QLabel("Hello",self)
        label.setFont(QFont("Arial",40))
        label.setGeometry(0,0,250,250)
        label.setStyleSheet("color:red;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # label.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignLeft)
        label2= QLabel(self)
        label2.setGeometry(0,0,250,250)
        pixmap=QPixmap("image.jpg")
        label2.setPixmap(pixmap)
        label2.setScaledContents(True)
        # label2.setGeometry(self.width()-label2.width()//2,self.height()-label2.height()//2,label2.width,label2.height())
        self.initUI()
    
    def initUI(self):
        
        self.line_edit.setGeometry(10,10,200,40)
        self.line_edit.placeholderText("Enter your name")
        self.line_edit.setStyleSheet("font-size:25px;""font-family:arial")
        self.button.setGeometry(210,10,100,40)
        self.button.setStyleSheet("font-size:25px;""font-family:arial")
        self.button.clicked.connect(self.submit)
        '''normally when creating widgets we prefix it with self followed by the  name 
        otherwise we will be creating local variables'''
       
        self.button.setGeometry(150,200,200,100)
        self.button.setStyleSheet("font-size:30px;")
        # a signal is ammitted when a widget is interacted wiht
        self.button.clicked.connect(self.on_click)
        self.button.setDisabled(True)
        
        def on_click(self):
            print("Button Clicked!")
            self.button.setText("Clicket!")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        label1 = QLabel("#1",self)
        label2 = QLabel("#2",self)
        label3 = QLabel("#3",self)
        label4 = QLabel("#4",self)
        label5 = QLabel("#5",self)
        
        label1.setStyleSheet("background-color:red")
        label2.setStyleSheet("background-color:pink")
        label3.setStyleSheet("background-color:purple")
        label4.setStyleSheet("background-color:coral")
        label5.setStyleSheet("background-color:blue")
        
        vbox= QVBoxLayout()
        vbox.addWidget(label1)
        vbox.addWidget(label2)
        vbox.addWidget(label3)
        vbox.addWidget(label4)
        vbox.addWidget(label5)
        
        central_widget.setLayout(vbox)
        
        #similar for horizonal layout just use qh layouts '
        #for grids you nedd grids and specifiy row,coloumn 
        """eg: grid=QgridLayout()
            grid.addwidget(labelx,row,coloumn)"""
    def submit(self):
         print("You clicked the button!")  
         text=self.line_edit.text()
         print(f"hello {text}")
        
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    
if __name__ == "__main__":
    main()            