#模板匹配
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def template_demo():
    tpl =cv.imread("headImg.png")
    target = cv.imread("korea womans.png")
    cv.namedWindow('template image', cv.WINDOW_NORMAL)
    cv.imshow("template image", tpl)
    cv.namedWindow('target image', cv.WINDOW_NORMAL)
    cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]   #3种模板匹配方法
    th, tw = tpl.shape[:2]
    i=0
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        i+=1
        plt.subplot(2,3,i)
        plt.title(str(md))
        plt.imshow(result,cmap='gray')
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)   #br是矩形右下角的点的坐标
        cv.rectangle(target, tl, (tl[0]+5, tl[1]+5) , (0, 0, 255), 2)
        cv.rectangle(target, tl, br, (0, 255, 255), 2)
        plt.subplot(2,3,i+3)
        plt.imshow(cv.cvtColor(target,cv.COLOR_RGB2BGR))
    plt.show()
        # cv.namedWindow("match-" + np.str(md), cv.WINDOW_NORMAL)
        # cv.imshow("match-" + np.str(md), target)

template_demo()
cv.waitKey(0)
cv.destroyAllWindows()