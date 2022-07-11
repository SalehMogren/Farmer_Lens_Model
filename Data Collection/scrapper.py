from time import sleep
import requests
from bs4 import BeautifulSoup
from itertools import zip_longest
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO
import random
import string
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from cloud_storage import upload_blob_from_memory
from tempfile import TemporaryFile, NamedTemporaryFile
BUCKET_NAME = 'dataset_collection'


def haraj_scrapper(query: str, folder_location: str):
    search = "تمر سكري مفتل"  # search whatever you want
    query.replace(' ', '%20')
    # url=f"https://haraj.com.sa/search/{search.split(' ')[0]}%20{search.split(' ')[1]}%20{search.split(' ')[2]}"
    url = f"https://haraj.com.sa/search/{query}"

    """
    url="https://haraj.com.sa/search/تمر%20سكري%20مفتل" 
    """

    """ downloading the main page """
    # browser = webdriver.Chrome(ChromeDriverManager().install())
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.implicitly_wait(5)
    browser.maximize_window()
    browser.get(url)
    sleep(5)
    bottom = browser.find_element(
        by=By.XPATH, value="//div[@class='footer_wrapper']")

    posts_num = 0

    """ going to bottom and pressing read more to view all the posts """
    while posts_num < 10:  # change to how many posts you want (chose a number that is a multiple of 20 )
        sleep(2)
        bottom.click()
        try:
            btn = browser.find_element(
                by=By.XPATH, value="//button[@id='more']")
            btn.click()
        except:
            bottom.click()
        posts_num += 20

    """
    now we start scraping posts, links and images

    """

    innerHTML = browser.execute_script("return document.body.innerHTML")

    soup = BeautifulSoup(innerHTML, "lxml")

    posts = soup.find_all('div', {'class': 'postTitle'})
    links = []
    images = []

    for post in posts:
        link = post.find('a').attrs['href']
        links.append('https://haraj.com.sa' + link)

    print('going to each link')

    for link in links:
        page = requests.get(link)
        # print(page.status_code) #has to be 200, if another number appeard that means that the script could not get inside the page
        soup = BeautifulSoup(page.content, "lxml")
        try:
            img = soup.find('span', {'class': 'img_wrapper'}).find(
                'img').attrs['srcset'].split()[2]
        except:
            img = None

        if img != None:
            images.append(img)

    print('------------------------')
    print(f"Haraj Scrapper summary fo {query}")
    print(f"Number of posts :{len(posts)}")
    print(f"Number of links :{len(links)}")
    # less then or equels the number of posts, becuse not all posts have images
    print(f"Number of image :{len(images)}")

    """ saving images """
    for img in images:
        response = requests.get(img)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_name = "".join(random.choice(string.ascii_lowercase)
                           for i in range(16)) + ".jpeg"

        bytes = BytesIO()
        img.save(bytes, 'jpeg')
        upload_blob_from_memory(
            BUCKET_NAME, bytes.getvalue(), f'{folder_location}/{img_name}')
