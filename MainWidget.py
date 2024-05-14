import os
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,\
    QComboBox, QLabel, QSpinBox, QFileDialog, QMessageBox
from PaintBoard import PaintBoard
from predicted import pre

class MainWidget(QWidget):


    def __init__(self, Parent=None):
        super().__init__(Parent)
        # 初始化
        self.__InitData()
        self.__InitView()
    
    def __InitData(self): # 初始化成员变量
        self.__paintBoard = PaintBoard(self)
        #获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames() 
        
    def __InitView(self): #  初始化界面
        self.setFixedSize(740,580)
        self.setWindowTitle("MNIST预测")
        
        main_layout = QHBoxLayout(self) 
        main_layout.setSpacing(10)   #设置主布局内边距以及控件间距为10px
        main_layout.addWidget(self.__paintBoard) 
        

        sub_layout = QVBoxLayout() 
        sub_layout.setContentsMargins(10, 10, 10, 10) 

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) 
        sub_layout.addWidget(self.__btn_Clear)
        
        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)
        
        self.__btn_Save = QPushButton("进行预测")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        splitter = QSplitter(self) #占位符
        sub_layout.addWidget(splitter)
    

        main_layout.addLayout(sub_layout) #将子布局加入主布局

    
    def on_btn_Save_Clicked(self):
        savePath = os.path.join("./img", "image.png")
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)
        # predicted
        res = pre()
        QMessageBox.information(self, "信息", "我想它应该是"+res)


        
    def Quit(self):
        self.close()
