from bs4 import BeautifulSoup
import re
def getInfo():
    file=open('city_code.html','rb')
    html=file.read()
    bs=BeautifulSoup(html,'html.parser')
    provinces=bs.find_all('province')
    citys=bs.find_all('county')
    city_infos={}
    for city in citys:
        attr=city.attrs
        city_infos[attr['name']]=attr['weathercode']
    return city_infos
