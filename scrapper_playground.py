import bs4
from bs4 import BeautifulSoup
from bs4 import NavigableString
import re

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""


def get_title_tag(soup: BeautifulSoup):
    print(soup.title)
    # <title>The Dormouse's story</title>

    print(soup.title.name)
    # u'title'

    print(soup.title.string)
    # u'The Dormouse's story'

    print(soup.title.parent.name)
    # u'head'


def get_p_tag(soup: BeautifulSoup):
    print(soup.p)
    # <p class="title"><b>The Dormouse's story</b></p>

    print(soup.p['class'])
    # u'title'


def get_a_tags(soup: BeautifulSoup):
    print(soup.a)
    # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

    print(soup.find_all('a'))
    # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

    print(soup.find(id="link3"))
    # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

    for link in soup.find_all('a'):
        print(link.get('href'))
    # http://example.com/elsie
    # http://example.com/lacie
    # http://example.com/tillie


def get_all_text(soup: BeautifulSoup):
    print(soup.get_text())
    # The Dormouse's story
    #
    # The Dormouse's story
    #
    # Once upon a time there were three little sisters; and their names were
    # Elsie,
    # Lacie and
    # Tillie;
    # and they lived at the bottom of a well.
    #
    # ...

    # can have almost the same output as above with stripped_string

    for string in soup.strings:
        print(repr(string))
    # u"The Dormouse's story"
    # u'\n\n'
    # u"The Dormouse's story"
    # u'\n\n'
    # u'Once upon a time there were three little sisters; and their names were\n'
    # u'Elsie'
    # u',\n'
    # u'Lacie'
    # u' and\n'
    # u'Tillie'
    # u';\nand they lived at the bottom of a well.'
    # u'\n\n'
    # u'...'
    # u'\n'


def get_sibkings(soup: BeautifulSoup):
    for sibling in soup.a.next_siblings:
        print(repr(sibling))
    # u',\n'
    # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
    # u' and\n'
    # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
    # u'; and they lived at the bottom of a well.'
    # None

    for sibling in soup.find(id="link3").previous_siblings:
        print(repr(sibling))
    # ' and\n'
    # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
    # u',\n'
    # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
    # u'Once upon a time there were three little sisters; and their names were\n'
    # None

    last_a_tag = soup.find("a", id="link3")
    print(last_a_tag)
    # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

    print(last_a_tag.next_sibling)
    # '; and they lived at the bottom of a well.'


def use_regex(soup: BeautifulSoup):
    # can use regex for filter searches
    for tag in soup.find_all(re.compile("^b")):
        print(tag.name)
    # body
    # b


def get_all_tags(soup: BeautifulSoup):
    for tag in soup.find_all(True):
        print(tag.name)
    # html
    # head
    # title
    # body
    # p
    # b
    # p
    # a
    # a
    # a
    # p


def has_class_but_no_id(tag):
    '''
    custom function example
    '''

    return tag.has_attr('class') and not tag.has_attr('id')


def not_lacie(href):
    return href and not re.compile("lacie").search(href)


def surrounded_by_strings(tag: bs4.element.ResultSet):
    return (isinstance(tag.next_element, NavigableString)
            and isinstance(tag.previous_element, NavigableString))


def filters(soup: BeautifulSoup):
    soup.find_all("title")
    # [<title>The Dormouse's story</title>]

    soup.find_all("p", "title")
    # [<p class="title"><b>The Dormouse's story</b></p>]

    soup.find_all("a")
    # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

    soup.find_all(id=True)
    # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

    soup.find_all(id="link2")
    # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

    soup.find_all(href=re.compile("elsie"))
    # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

    soup.find(string=re.compile("sisters"))
    # u'Once upon a time there were three little sisters; and their names were\n'

    soup.find_all(href=re.compile("elsie"), id='link1')
    # [<a class="sister" href="http://example.com/elsie" id="link1">three</a>]


def navigate_html(soup: BeautifulSoup):
    # for i in soup.find_all(has_class_but_no_id):
    #     print(i)

    return


def main():
    # html_doc_2 = open("./soup_html.html", "r").read()
    # or
    # with open("./soup_html.html", "r") as fp:
    # soup = BeautifulSoup(fp)

    # in case of external lib, requires `lxml` or `html5lib` deps
    soup = BeautifulSoup(html_doc, 'lxml')

    # pretty html string
    # print(soup.prettify())

    navigate_html(soup)


main()
