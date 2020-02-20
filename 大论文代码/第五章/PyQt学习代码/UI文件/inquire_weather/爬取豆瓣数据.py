import requests
import os
from bs4 import BeautifulSoup
with open("douban.txt", "w",encoding='utf-8') as f:
    url = 'https://www.douban.com/'
    headers={ "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"}
    data = requests.get(url,headers=headers)
    soup=BeautifulSoup(data.text,'html.parser')
    # print(soup)
    f.write(data.text)
    book_left=soup.find(('ul',{'class':'cover-col-4 clearfix'}))
    book_left=book_left.find_all('li')
    print(book_left)