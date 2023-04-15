import mechanicalsoup
from bs4 import BeautifulSoup

# globals

base_url = "http://olympus.realpython.org"


def main():
    # get link with form
    url = f"{base_url}/login"
    browser = mechanicalsoup.Browser()
    login_page = browser.get(url)
    login_html = login_page.soup

    # input values in form
    form = login_html.select("form")[0]
    form.select("input")[0]["value"] = "zeus"
    form.select("input")[1]["value"] = "ThunderDude"

    # update page with form and get new page
    profiles_page = browser.submit(form, login_page.url)

    # get links
    links = profiles_page.soup.select("a")

    for link in links:
        address = base_url + link["href"]
        text = link.text
        print(f"{text}: {address}")


main()
