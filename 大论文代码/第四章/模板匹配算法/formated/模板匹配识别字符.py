import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
name_list='0，1，2，4，6，7，8，9，A，B'.split('，')

def detect_code(img,show_plt=False):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    info_dict=[]
    global_min_value=10.0
    global name_list
    global_code=None
    for i in range(len(name_list)):
        target=cv2.imread('formatedImg\\'+name_list[i]+'.jpg',cv2.IMREAD_GRAYSCALE)
        #TM_SQDIFF_NORMED完全匹配返回的是0
        result = cv2.matchTemplate(target, img, cv2.TM_SQDIFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        th, tw = img.shape[:2]
        cv2.rectangle(target, min_loc, (min_loc[0] + tw, min_loc[1] + th), (0, 0, 0), 1)
        # plt.subplot(2,5,i+1)
        # plt.title('匹配值：'+str(min_val),fontsize=15)
        # plt.imshow(target,cmap='gray')
        if(i==0):
            global_min_value=min_val
            global_code = name_list[i]

        if(min_val<global_min_value):
            global_min_value=min_val
            global_code=name_list[i]
    if(show_plt==True):
        plt.show()
    return global_code

if __name__=='__main__':
    src=cv2.imread('cutedImg\\A.jpg',cv2.IMREAD_GRAYSCALE)
    # ret, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('name',src)
    ret_code=detect_code(src,True)
    print(ret_code)
    cv2.waitKey()