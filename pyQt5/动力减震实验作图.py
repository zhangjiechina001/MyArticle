import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
fontsize=15
x=[37,47,56,190,362,528]
y=[0.978,0.233,0.219,0.032,0.021,0.015]
plt.plot(x,y,'g--o',label='隔振系数',color='r')
plt.legend(loc='high right',fontsize=fontsize)
plt.xlabel('激励频率(Hz)',fontsize=fontsize)
plt.ylabel('隔振系数',fontsize=fontsize)
plt.title('')
plt.show()