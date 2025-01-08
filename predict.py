#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from archs import UNext
from PIL import Image
import os
import time
UNext = UNext(1)
imgs = os.listdir("./img")

for jpg in imgs:
    

    img = Image.open("./img/"+jpg)
    start_time = time.time()
    image = UNext.detect_image(img)
    duration = time.time() - start_time
    print("预测时间",duration)
    image.save("./img_out/"+jpg)
    


# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = pspnet.detect_image(image)
#         r_image.show()
