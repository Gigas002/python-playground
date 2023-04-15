from bs4 import BeautifulSoup
from urllib.request import urlopen

def get_image_tags(soup : BeautifulSoup):
    image1, image2 = soup.find_all("img")

    print(image1["src"])
    print(image2["src"])

def main():
    url = "http://olympus.realpython.org/profiles/dionysus"
    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "lxml")

    get_image_tags(soup)

main()
