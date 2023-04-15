from urllib.request import urlopen
import re

def standard_extract_from_html(html : str):
    # index of title tag
    title_index = html.find("<title>")

    # index of contents itself
    start_index = title_index + len("<title>")
    end_index = html.find("</title>")

    # extract the title, using indexes
    title = html[start_index:end_index]

    print(title)

def re_extract_from_html(html : str):
    pattern = "<title.*?>.*?</title.*?>"
    match_results = re.search(pattern, html, re.IGNORECASE)
    title = match_results.group()
    title = re.sub("<.*?>", "", title) # Remove HTML tags

    print(title)

def extract_dionysus(html : str):
    name_index = html.find("<h2>")
    start_index = name_index + len("<h2>")
    end_index = html.find("</h2>")

    name = html[start_index:end_index]

    color_index = html.find("<br>\n<br>\n")
    start_index = color_index + len("<br>\n<br>\n")
    end_index = html.find("\n</center>")

    color = html[start_index:end_index]

    print(color)

    # print(name)

def extract_dionysus_loop(html : str):
    for string in ["Name: ", "Favorite Color:"]:
        string_start_idx = html.find(string)
        text_start_idx = string_start_idx + len(string)

        next_html_tag_offset = html[text_start_idx:].find("<")
        text_end_idx = text_start_idx + next_html_tag_offset

        raw_text = html[text_start_idx : text_end_idx]
        clean_text = raw_text.strip(" \r\n\t")

        print(clean_text)

def main():
    # url = "http://olympus.realpython.org/profiles/aphrodite"
    # url = "http://olympus.realpython.org/profiles/poseidon"
    url = "http://olympus.realpython.org/profiles/dionysus"

    # read the page as http.client.HTTPResponse object 
    page = urlopen(url)

    # get html page from response object
    html = page.read().decode("utf-8")

    # standard_extract_from_html(html)
    # re_extract_from_html(html)
    # extract_dionysus(html)
    extract_dionysus_loop(html)

main()
