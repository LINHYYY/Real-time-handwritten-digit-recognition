from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt

class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)

        self.__InitData() #初始化
        self.__InitView()
        
    def __InitData(self):
        self.__size = QSize(480,460)

        self.__board = QPixmap(self.__size)
        self.__board.fill(QColor(68,1,84)) #用紫色填充画板，与数据集保持一致rgb(68,1,84)
        
        self.__IsEmpty = True
        self.EraserMode = False
        self.__lastPos = QPoint(0,0 )#上一次鼠标位置
        self.__currentPos = QPoint(0,0) #当前的鼠标位置
        
        self.__painter = QPainter()
        self.__thickness = 30  #画笔粗细
        self.__penColor = QColor(QColor(246,230,31))  #设置画笔颜色rgb(246,230,31)

     
    def __InitView(self):
        self.setFixedSize(self.__size)
        
    def Clear(self):
        #清空画板
        self.__board.fill(QColor(68,1,84)) # rgb(68,1,84)
        self.update()
        self.__IsEmpty = True
        
    def ChangePenThickness(self, thickness=10):
        self.__thickness = thickness
        
    def IsEmpty(self):
        #返回画板是否为空
        return self.__IsEmpty
    
    def GetContentAsQImage(self):
        #获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image
        
    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()
        
    def mousePressEvent(self, mouseEvent):
        #鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos =  mouseEvent.pos()
        self.__lastPos = self.__currentPos
        
        
    def mouseMoveEvent(self, mouseEvent):
        #鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos =  mouseEvent.pos()
        self.__painter.begin(self.__board)
        
        self.__painter.setPen(QPen(self.__penColor,self.__thickness))
            
        #画线    
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos
                
        self.update() #更新显示
        
    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False #画板不再为空
        
