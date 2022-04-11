import re
import time
import os

from bs4 import BeautifulSoup as soup
import requests

url = 'https://leagueoflegends.fandom.com/wiki/List_of_champions/Base_statistics'
base_url = 'http://leagueoflegends.fandom.com'


html = requests.get(url).text
sp = soup(html)
table = sp.find('table')
imgs = table.find_all('img')

img_links = [i['src'] for i in imgs]
img_links += [i['data-src'] for i in imgs if i.has_attr("data-src")]
img_links = [i[:i.find("/revision")] for i in img_links if "OriginalCircle.png" in i]



if not os.path.exists("gendataset/champion_icons/"):
    os.makedirs("gendataset/champion_icons/")


for img_link in img_links:
    pattern = r"\/.+\/(.+)_Original"
    champ_name = re.findall(pattern, img_link)[0]
    with open(f'./gendataset/champion_icons/{champ_name}.jpg','wb') as f:
        content = requests.get(img_link).content
        f.write(content)
    print(img_link, champ_name)