# 导入工具包
import numpy as np
import argparse
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())


# 对矩形的4个点按一定顺序排序
def order_points(pts):
    print(pts)
    # 一共4个坐标点,创建一个形状为(4, 2)的零矩阵rect，dtype = "float32"表示该矩阵的数据类型是32位浮点数。
    #  这个矩阵最终将存放排序后的四个点的坐标。
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下,利用横纵坐标相加，和最小的为左上，最大的为右下
    s = pts.sum(axis=1)  # x+y求和
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下，利用y-x,找到右上角和左下角，右上角是y-x最小的点，左下角是y-x最大的点
    diff = np.diff(pts, axis=1)
    print(diff)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    print(rect)
    # 返回按顺序排完后的点组成的4*2矩阵
    return rect


# 对图像进行透视变换，给定四个点的坐标作为变换的参考。校正图像中的矩形对象，使矩形看起来更直，
def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 使用原始的四个点 (rect) 和变换后的点 (dst) 计算透视变换矩阵 M。
    M = cv2.getPerspectiveTransform(rect, dst)
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 最终会存储调整大小后的图像的新维度。
    dim = None
    (h, w) = image.shape[:2]
    # 如果没有指定新的宽度和高度，就会直接返回原图像。
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # cv2.INTER_AREA 是一个在缩小时使用的插值方法，能够提供高质量的结果。
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

print("STEP 1: 读入并放大两倍")
# 读取输入
image = cv2.imread(args["image"])
# 坐标也会相同变化,先记录缩放的比列，便于后续恢复
ratio = 2
orig = image.copy()
# 将图像放大，以应对文字密集或文字太小，导致影响后续阈值处理
image = resize(orig, height=image.shape[0]*ratio,inter =cv2.INTER_LINEAR)

# 转为HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",resize(hsv_image,height=800))
# 定义白色在HSV空间的范围
lower_white = np.array([0, 0, 120], dtype=np.uint8)
upper_white = np.array([180, 60, 255], dtype=np.uint8)

# 创建一个掩码，过滤出白色区域
mask = cv2.inRange(hsv_image, lower_white, upper_white)
cv2.imwrite('mask.jpg', mask)
# 根据这个掩码区域，找出纸张的轮廓
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓，找出文本区域
for c in contours:
    # 计算轮廓，C表示输入的点集，True表示封闭的
    peri = cv2.arcLength(c, True)
    # 对轮廓点集进行近似，使其减少点的数量，变成一个近似的多边形。0.02 * peri 是近似度参数，这个值越大，结果轮廓的角点越少，即多边形越简单。
    # True 参数指出这是一个闭合轮廓。
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当多边形只有四个点时就认为找到了文本区域
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("STEP 2: 获取轮廓")
# 用于调试
draw = image.copy()
# 在原始图像image上绘制之前被检测出来的文本区域，-1绘制所有轮廓（这里只有一个轮廓），2是轮廓的宽度
cv2.drawContours(draw, [screenCnt], -1, (0, 255, 0), 8)

# 透视变换，将刚才检测出来的本文区域矫正为一个正的矩形试图
warped = four_point_transform(image, screenCnt.reshape(4, 2))
cv2.imshow("Draw",resize(draw,height=800))
cv2.imshow('transform', resize(warped,height=800))
cv2.waitKey(0)
cv2.destroyAllWindows()


# 变换后的图像转为灰度图像-》二值化阈值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.adaptiveThreshold(
    warped,  # 灰度图像
    255,  # 超过阈值后赋予的值
    cv2.ADAPTIVE_THRESH_MEAN_C,  # 使用均值作为阈值计算的方法
    cv2.THRESH_BINARY,  # 阈值类型
    19,  # 邻域大小（blockSize）
    3  # 从均值中减去的常数，微调阈值
)
cv2.imwrite('REF.jpg', ref) # 用于调试
ref = cv2.medianBlur(ref,13)
cv2.imwrite('scan.jpg', ref)

# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height=800))
cv2.imshow("Scanned", resize(ref, height=800))
cv2.waitKey(0)
