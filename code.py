import cv2
import paddlehub as hub
import pandas as pd
import numpy as np
from PIL import ImageFont,ImageDraw,Image

# 定义参数
# define some parameters
img_dir='案例.jpg'             # 打卡截图
name_csv='案例.csv'     # 花名册文件，表头为(name,1,2,3,4,5,6,7...)，对应的内容分别为姓名，第一天的打卡结果，第二天的打卡结果...，其中姓名填入即可，其他方格可以留空
save_dir='输出案例.csv'       # 打卡后的结果保存路径
block_size=25               # 右上角的绿色对勾的区域大小
w_bias=35                   # 绿色对勾相对于姓名位置的横向偏移量
h_bias=-50                  # 绿色对勾相对于姓名位置的纵向偏移量
threshold=175               # 阈值，低于此阈值则为绿色，否则则偏向于灰色，可以根据像素点均值进行分类
clock_in_col=1              # 打卡条目，此处设为1对应第一天的打卡结果

# 导入文字识别模型
# ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
ocr = hub.Module(name="chinese_ocr_db_crnn_server")

# 读取测试文件夹test.txt中的照片路径
img=cv2.imread(img_dir)
np_images =[img] 

results = ocr.recognize_text(
                    images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=True,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.8,           # 检测文本框置信度的阈值；
                    text_thresh=0.8)          # 识别中文文本置信度的阈值；

# 匹配文字与识别结果
# the function make codes shorter
def is_name_in_ocr(name,ocr_result):
    for item in ocr_result[0]['data']:
        if item['text']==name:
            return True, item['text_box_position']
    return False,None

# 根据指定文字生成图片并用ocr模型识别
def generate_name_and_ocr(name,ocr):
    w=200
    h=100
    img = np.full((h,w,3),fill_value=255,dtype=np.uint8)
    fontpath = "simhei.ttf" #导入字体文件
    b,g,r,a = 0,0,0,0 #设置字体的颜色
    font = ImageFont.truetype(fontpath,30)#设置字体大小
    img_pil = Image.fromarray(img)#将numpy array的图片格式转为PIL的图片格式
    draw = ImageDraw.Draw(img_pil)#创建画板
    draw.text((w/2-10,h/2-20),name,font=font,fill=(b,g,r,a))#在图片上绘制中文
    img = np.array(img_pil)#将图片转为numpy array的数据格式
    results = ocr.recognize_text(
                    images=[img],         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                    use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                    output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                    visualization=True,       # 是否将识别结果保存为图片文件；
                    box_thresh=0.8,           # 检测文本框置信度的阈值；
                    text_thresh=0.8)          # 识别中文文本置信度的阈值；
    return results[0]['data'][0]['text']

# 检索识别框右上角的绿色方块，如果存在则打卡成功
df=pd.read_csv(name_csv,encoding='gbk')
for i in range(len(df)):
    name=df.iloc[i,0]
    name=generate_name_and_ocr(name,ocr) # 注释此行代码可以看到 案例 中的佘大同学签到失败了
    flag, text_box=is_name_in_ocr(name,results)
    if flag:
        center_point=[(text_box[2][0]-text_box[0][0])/2+text_box[0][0],(text_box[2][1]-text_box[0][1])/2+text_box[0][1]]
        center_point=[int(x) for x in center_point]
        check_img=img[center_point[1]+h_bias:center_point[1]+h_bias+block_size,center_point[0]+w_bias:center_point[0]+w_bias+block_size,:]
        # print(name,check_img.mean())
        if check_img.mean()<threshold:
            df.iloc[i,clock_in_col]=1
df.to_csv(save_dir,index=None)
