import urllib
from urllib import parse
import base64
import cv2
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}
postData = {
    'image' : base64.b64encode(cv2.imread('.jpg', cv2.imread('formatedImg\\9.jpg'))[1]).decode(), #图片 base64格式
    'language_type': 'CHN_ENG', #中英双语
    'detect_direction': 'false', #文字方向
    'detect_language': 'false', #语言
    'probability': 'true', #判断为目标文字的概率
}
data = urllib.parse.urlencode(postData).encode('utf-8')
req = urllib.request.Request(url='https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?'
                           'access_token=**********************',
                           headers=headers, data = data)
response = urllib.request.urlopen(req)
# 通过get请求返回的文本值
print(response.read().decode('utf-8'))
