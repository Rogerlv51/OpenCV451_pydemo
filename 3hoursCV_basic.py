
from inspect import stack
import cv2
from cv2 import COLOR_BGR2HSV
from cv2 import getTrackbarPos
import numpy as np

# 读取视频
def readVideo():
    video = cv2.VideoCapture('Resources/test_video.mp4')

    while True:
        success, img = video.read()   # success是返回一个bool值判断视频是否导入成功
        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # 一秒播放一帧图片，按下q键退出
            break


# 调取摄像头
def capCamera():
    video = cv2.VideoCapture(0)
    video.set(3, 640)  # 设置窗口的宽，第一个参数是设置的一些id号表示
    video.set(4, 480)  # 设置窗口的高
    video.set(10, 100)  # 设置画面亮度

    while True:
        success, img = video.read()   # success是返回一个bool值判断视频是否导入成功
        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):   # 一秒播放一帧图片，按下q键退出
            break


# 膨胀操作
def img_Dialation(path):   # 传一个图片地址，字符串
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny边缘检测
    img_canny = cv2.Canny(img_gray, 50, 180) # threshold1和threshold2之间的最小值用于边缘连接,最大值用于寻找强边界与背景分离
    kernel = np.ones((5,5), dtype=np.uint8)   # 5×5的卷积核
    img_dilation = cv2.dilate(img_canny, kernel, iterations=1)  # 可以指定迭代次数即膨胀几次
    cv2.imshow('canny', img_canny)
    cv2.imshow('dilation', img_dilation)
    cv2.waitKey()


# Resize操作和图像剪切
def img_ResizeAndCrop(path):
    img = cv2.imread(path)
    print(img.shape)   # 这里输出的shape是高在前宽在后的，因为np矩阵是先行后列
    img_resize = cv2.resize(img, (300, 200))  # 宽在前高在后
    print(img_resize.shape)
    img_croped = img[:200, :300]   # 截取图片
    cv2.imshow('img', img)
    cv2.imshow('resize', img_resize)
    cv2.imshow('croped', img_croped)
    cv2.waitKey()


# 在图片上画各种框和添加文字信息
def boxAndText():
    img = np.zeros((512,512,3), np.uint8)
    # img[:] = 0, 0, 255   # 可以给原图上色，这个bgr值显然是红色
    cv2.line(img, (0,0), (255,255), (255,0,0), 3)  # 画直线
    cv2.rectangle(img, (50,50), (400,400), (0,255,0), 2)  # 线宽参数为负数时表示填充
    cv2.circle(img, (255,255), 50, (255,255,0), 5)
    # 设置文字位置，字体样式大小即颜色
    cv2.putText(img, 'OPENCV ', (220, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    print(img)
    cv2.imshow('img', img)
    cv2.waitKey()


# 透视变换
def WarpTrans(path):
    img = cv2.imread(path)
    weight, height = 250, 350   # 分别从一个图像的左上，右上，左下，右下读取坐标，先宽后高
    pts1 = np.float32([[111,219], [287,188], [154,482], [352,440]])
    pts2 = np.float32([[0,0], [weight,0], [0,height], [weight,height]])
    matraix = cv2.getPerspectiveTransform(pts1,pts2)
    img_out = cv2.warpPerspective(img, matraix,(weight,height))
    cv2.imshow('src', img)
    cv2.imshow('out', img_out)
    cv2.waitKey()


# 把图像组合到一起显示输出
def showDiffImg(path):
    img = cv2.imread(path)
    out1 = np.hstack((img, img))  # 水平组合
    out2 = np.vstack((img, img))  # 垂直组合
    cv2.imshow("hstack", out1)
    cv2.imshow("vstack", out2)
    cv2.waitKey()


# 以任意形式组合图片的万能函数，可以规定图片的显示尺寸
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



# 利用trackbar来做颜色检测提取
def color_Detect():

    # 回调函数onchange
    def onchange(value):
        while True:
            hin_min = getTrackbarPos("Hue min", "TrackbarWin")
            hin_max = getTrackbarPos("Hue max", "TrackbarWin")
            sat_min = getTrackbarPos("Sat min", "TrackbarWin")
            sat_max = getTrackbarPos("Sat max", "TrackbarWin")
            val_min = getTrackbarPos("Val min", "TrackbarWin")
            val_max = getTrackbarPos("Val max", "TrackbarWin")
            print(hin_min, hin_max, sat_min, sat_max, val_min, val_max)
            lower = np.array([hin_min, sat_min, val_min])
            upper = np.array([hin_max, sat_max, val_max])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgResult = cv2.bitwise_and(img, img, mask=mask)  # 把颜色取出来显示在原图上
            imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
            cv2.imshow("imgStack", imgStack)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    

    path = 'Resources/lambo.png'
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, COLOR_BGR2HSV)
    cv2.namedWindow("TrackbarWin")
    cv2.resizeWindow("TrackbarWin", 640, 240)
    # 所有导航条必须依附于一个windows窗口中
    cv2.createTrackbar("Hue min", "TrackbarWin", 0, 179, onchange)
    cv2.createTrackbar("Hue max", "TrackbarWin", 179, 179, onchange)
    cv2.createTrackbar("Sat min", "TrackbarWin", 0, 255, onchange)
    cv2.createTrackbar("Sat max", "TrackbarWin", 255, 255, onchange)
    cv2.createTrackbar("Val min", "TrackbarWin", 0, 255, onchange)
    cv2.createTrackbar("Val max", "TrackbarWin", 255, 255, onchange)










if __name__ == '__main__':
    # readVideo()
    # img_Dialation('Resources/lena.png')
    # img_ResizeAndCrop('Resources/lambo.PNG')
    # boxAndText()
    # WarpTrans('Resources/cards.jpg')
    # showDiffImg('Resources/lena.png')
    color_Detect()
