from bs4 import BeautifulSoup
import requests as res #導入requests library
import re
import time
from multiprocessing import Pool
import multiprocessing as mp

print('蘋果今日焦點')

def clawer(url):
    
    headers = {'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'authority': 'tw.appledaily.com',
        'accept':'text/html,application/xhtml+xml,application/xml;'
        'q=0.9,image/webp,image/apng,*/*;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        '(KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36'
        }
    s = res.session() #存取cookies
    resp = s.get(url,headers=headers).text
    soup = BeautifulSoup(resp, 'html5lib')

    ############### 清理標題 ############
    new_title = []

    titles = soup.find_all('div',{'class':"abdominis rlby clearmen"})
    for s in titles:
        s = s.find_all("h1")
    for i in s:
        i = i.text
        new_title.append(i)

    title = new_title[1:]

    ############### 清理連結 ############
    for q in titles:
        q = q.find_all(href=re.compile("^.*realtime\/\d{8}\/\d{7}\/$"))
        hrefs = q 

    link=[]
    for d in hrefs:
        href = d.get('href')
        link.append(href)

    ############### 建立 Json format ############
    article = []
    for title,link in zip(title,link):

        article.append({
            "標題":title,
            "連結":link,
        })

    for i in article:
        print(i)

if __name__ == "__main__" :
    t1 = time.time()
    urls=[]
    for i in range(1,30):
        v = str(i)
        url = 'https://tw.appledaily.com/new/realtime/'+v
        urls.append(url)
    pool = Pool() # Pool() 不放參數則默認使用電腦核的數量
    pool.map(clawer,urls) 
    pool.close()
    pool.join()

    print('Total time: %.1f s' % (time.time()-t1))  
