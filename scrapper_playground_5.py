import mechanicalsoup
import time

# globals
base_url = "http://olympus.realpython.org"

def main():
    browser = mechanicalsoup.Browser()

    while True:
        page = browser.get(f"{base_url}/dice")
        tag = page.soup.select("#result")[0]
        result = tag.text

        print(f"The result of your dice roll is: {result}")
        
        is_continue = input("Continue? Y/n: ")

        if (is_continue == "n"):
            break

        print("Please wait...")
        time.sleep(10)


main()
