import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
# 这几个包是用来实时显示UI界面方便我们调整参数的


'''
    当你要导入某个模块，但又不想改模块的部分代码被直接执行
    那就可以这一部分代码放在“if __name__=='__main__':”内部
'''


def image_inverse(img):   # 灰度反色变换，灰度值反转
    value_max = np.max(img)
    img_out = value_max - img
    return img_out


def image_log(img):   # 灰度对数变换
    img_out = np.log(1+img)
    return img_out


def gamma_trans(input, gamma=2, eps=0):  # gamma变换
    return 255. * (((input + eps)/255.) ** gamma)   # eps是一个补偿因子，对灰度值为0的地方进行补偿


def update_gamma(val):   # 回调函数接收滑块传的值
    gamma = slider1.val
    output = gamma_trans(gamma_img, gamma=gamma, eps=0.5)
    print("----\n", output)
    plt.title("伽马变换后，gamma = " + str(gamma))
    plt.imshow(output, cmap='gray', vmin=0, vmax=255)


if __name__ == '__main__':
    gray_img = np.asarray(Image.open('Images/X.jpg').convert('L'))  # 以灰度值方式读取
    inv_img = image_inverse(gray_img)

    gray_img2 = np.asarray(Image.open('Images/squared_paper.jpg').convert('L'))
    gamma_img = np.asarray(Image.open('Images/jisoo1.jpg').convert('L'))


    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 要是图像标题想取中文则要加上下面两行代码，pycharm中就这样别问为啥
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(121)
    plt.title('原图')
    plt.imshow(gamma_img, cmap='gray', vmin=0, vmax=255)   # 控制灰度值范围0-255

    
    plt.subplots_adjust(bottom=0.3)   # 设置滑块位置为画布底部30%处
    s1 = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    # 滑块范围0-2
    slider1 = Slider(s1, '参数gamma', 0.0, 2.0, valfmt='%.f', valinit=1.0, valstep=0.1)
    plt.subplot(122)
    slider1.on_changed(update_gamma)   # 回调函数来响应滑动滑块的值
    slider1.reset()
    slider1.set_val(1)

    plt.show()


    
