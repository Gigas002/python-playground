# Binary for chromedriver can be downloaded here:
# https://chromedriver.chromium.org/downloads
# Be sure to select the right version for you chrome installation

# imports

import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By
import time
import sys
import getopt
from lxml import html
from datetime import datetime

# global variables
base_url = "https://tenki.jp"


def init_chromedriver(path: str):
    options = Options()
    chrome_service = fs.Service(executable_path=path)

    return webdriver.Chrome(options=options, service=chrome_service)


def get_page_urlopen(url: str):
    page = urlopen(url)

    return page.read().decode("utf-8")


def serialize_to_csv(date: str, weather: str, t_max: str, t_min: str,
                     rain_possibilities: str, datetime: str):
    # date_series = pd.Series(date, name="日付")
    # weather_series = pd.Series(weather, name="天気")
    # t_max_series = pd.Series(t_max, name="最高気温")
    # t_min_series = pd.Series(t_min, name="最低気温")
    # rain_possibilities_series = pd.Series(rain_possibilities, name="降水確率")

    data = {
        "日付": [date],
        "天気": [weather],
        "最高気温": [t_max],
        "最低気温": [t_min],
        "降水確率": [rain_possibilities]
    }

    df = pd.DataFrame(data)

    # df = pd.concat([date_series, weather_series, t_max_series,
    #                 t_min_series, rain_possibilities_series], axis=1)
    df.to_csv(f"tenki_{datetime}.csv", encoding="utf-8_sig", index=False)


def get_page_with_selenium(driver: webdriver.Chrome, post_index: str = "104-0061"):
    driver.get(base_url)
    driver.implicitly_wait(10)

    # send data into textbox
    driver.find_element(
        By.XPATH, "/html/body/header/div[4]/div[1]/form/input").send_keys(post_index)

    # push the button
    driver.find_element(
        By.XPATH, "/html/body/header/div[4]/div[1]/form/button").click()

    # wait for page to load
    time.sleep(2)

    # select location
    driver.find_element(
        By.XPATH, "/html/body/div[2]/section/section[1]/div[2]/p/a").click()

    # wait for page to load
    time.sleep(2)

    # get current date
    date = driver.find_element(
        By.XPATH, "/html/body/div[3]/section/div[6]/div[1]/section/div/div[2]/div/p").text

    # get current weather
    weather = driver.find_element(
        By.XPATH, "/html/body/div[3]/section/div[4]/section[1]/div[1]/div[1]/p").text

    # get max temperature
    t_max = driver.find_element(
        By.XPATH, "/html/body/div[3]/section/div[4]/section[1]/div[1]/div[2]/dl/dd[1]/span[1]").text

    # get min temperature
    t_min = driver.find_element(
        By.XPATH, "/html/body/div[3]/section/div[4]/section[1]/div[1]/div[2]/dl/dd[3]/span[1]").text

    # get rain possibilities's table
    table = driver.find_element(
        By.XPATH, "/html/body/div[3]/section/div[4]/section[1]/div[2]/table/tbody").text

    # calculate indexes of start and end to search in resulting string
    s_idx = table.find("降水確率 ")
    e_idx = table.find("風 ")

    # get rain possibilities by index in string
    rain_possibilities = table[s_idx + 5:e_idx].strip()

    # print out ready data
    print(f"日付:{date}")
    print(f"天気:{weather}")
    print(f"最高気温:{t_max}")
    print(f"最低気温:{t_min}")
    print(f"降水確率:{rain_possibilities}")

    return (date, weather, t_max, t_min, rain_possibilities)


def parse_page_with_soup(soup: BeautifulSoup):
    # date
    date = soup.find_all("h3", "left-style")[0].text
    s_idx = date.find("今日")
    e_idx = date.rfind("日")
    date = date[s_idx + 3:e_idx + 1]

    # weather and min/max temps
    weather = soup.find_all("p", "weather-telop")[0].text
    t_max = soup.find_all("dd", "high-temp temp")[0].text
    t_min = soup.find_all("dd", "low-temp temp")[0].text

    # extract table and rain possibilities
    table = soup.select('table')[0]
    table_df = pd.read_html(str(table))[0]
    r_pos = table_df.loc[0].tolist()[1:]
    rain_possibilities = f"{r_pos[0]} {r_pos[1]} {r_pos[2]} {r_pos[3]}"

    # print out ready data
    print(f"日付:{date}")
    print(f"天気:{weather}")
    print(f"最高気温:{t_max}")
    print(f"最低気温:{t_min}")
    print(f"降水確率:{rain_possibilities}")

    return (date, weather, t_max, t_min, rain_possibilities)


# main func


def main(argv):
    '''
    Run example (doesn't use chromedriver, parse the hardcoded link using urlopen and soup):\n
    python ./kadai_tenki.py\n
    or parse by using selenium (chromedriver.exe path required):\n
    python ./kadai_tenki.py -c "chromedriver.exe"
    '''
    
    # init command line args

    opts, args = getopt.getopt(argv, "hc:", ["chromedriver="])
    chromedriver_path = ''
    is_driver = False

    for opt, arg in opts:
        if opt in ("-c", "--chromedriver"):
            chromedriver_path = arg
            is_driver = True
        elif opt in ("-h", "--help"):
            print("python ./kadai_tenki.py -c \"chromedriver.exe\"")
            sys.exit()
        else:
            is_driver = False

    # run code after args check

    driver: webdriver.Chrome
    date = ''
    weather = ''
    t_max = ''
    t_min = ''
    rain_possibilities = ''

    # using selenium and manual element selection by mouse

    if (is_driver):
        driver = init_chromedriver(chromedriver_path)
        (date, weather, t_max, t_min,
         rain_possibilities) = get_page_with_selenium(driver)

    # using BeautifulSoup (not ideal...)
    # It would be good to get XPath from BeautifulSoup object, but
    # it seems there's no way to do so

    if (not is_driver):
        url = f"{base_url}/forecast/3/16/4410/13102/"
        html = get_page_urlopen(url)
        soup = BeautifulSoup(html, "lxml")
        (date, weather, t_max, t_min, rain_possibilities) = parse_page_with_soup(soup)

    # serialize ready data to csv

    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")

    serialize_to_csv(date, weather, t_max, t_min,
                     rain_possibilities, date_time)

    # properly close driver at the end of work
    if (is_driver):
        # uncomment for debugging
        # input("Press any button to close the app")
        driver.close()


if __name__ == "__main__":
    main(sys.argv[1:])
