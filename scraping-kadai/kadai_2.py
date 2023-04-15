# imports

import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import sys
import getopt
from datetime import datetime
import smtplib
import ssl
import getpass
import random

# global variables

base_url = "https://alert.shop-bell.com/books/title_list/%E3%81%82/"

port = 465

# Works only with gmail sender account

smtp_server = "smtp.gmail.com"

# Message template with subject

base_message = "\nSubject: Test\n\n"


def get_page_urlopen(url: str):
    page = urlopen(url)

    return page.read().decode("utf-8")


def parse_page_with_soup(soup: BeautifulSoup, verbose: bool):
    # find table
    table = soup.select('table')[0]
    table_df = pd.read_html(str(table))[0]

    if (verbose):
        print(table_df)

    return table_df[["連載タイトル", "最新刊", "次巻"]]


def write_html(html: str, name: str = "kadai_2"):
    '''
    Useful for debugging purposes
    '''

    f = open(f"{name}.html", "w", encoding="utf-8")
    f.write(html)
    f.close


def save_to_csv(df):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")
    df.to_csv(f"{date_time}.csv", encoding="utf-8_sig", index=False)


def is_opts_failed(sender_email, reciever_email, init_failed):
    '''
    Check if options were initialized correctly
    '''

    if init_failed:
        return True
    elif not sender_email or not reciever_email:
        return True


def get_random_manga(df: pd.DataFrame):
    random_manga_id = random.randrange(0, len(df))

    return df.iloc[[random_manga_id]]


def row_to_message(df_row: pd.DataFrame):
    title = df_row.iloc[0, 0]
    latest_publication = df_row.iloc[0, 1]
    next_volume = df_row.iloc[0, 2]

    message = f"Title:{title}\nLatest publication:{latest_publication}\nNext volume:{next_volume}"

    return base_message + message


def send_email(sender_email, reciever_email, message, try_send):
    '''
    Set `try_send` variable to `True` for sending attempt. Otherwise will just print
    mailing data in command line.
    Note: This action requires google account and allowance of "less secure" apps in gmail settings;
    otherwise, google will throw an error on autherization attempt
    '''

    password = getpass.getpass()

    # create a secure SSL context
    context = ssl.create_default_context()

    if (try_send):
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, reciever_email, message)
    else:
        print(f"From:{sender_email}\nTo:{reciever_email}\nContent:\n{message}")


# main func

def main(argv):
    '''
    Options explanation:
    -v/--verbose: controls the output of DataFrame in command line. Default value is "False"
        If set, the DataFrame content of the page will be printed out in command line
    -t/--try-send: controls the email sending function. Default value is "False"
        If set, the program will attempt to actually send an email. This requires
        using gmail account with allowance of "less secure" apps in gmail settings.
        By default will just print an emailing info into command line
    -s/--sender-email: author of email
        There's no check for input string, so if you're not willing to actually send an email
        you can just set it to any word, e.g.: "Foo"
    -r/--reciever-email: reciever of email
        Applies the same rules as for `sender-email` variable
    '''

    opts, args = getopt.getopt(argv, "vts:r:", ["verbose", "try-send",
                                                "sender-email=", "reciever-email="])
    verbose = False
    try_send = False
    sender_email = ""
    reciever_email = ""
    init_failed = False

    if (len(opts) <= 0):
        init_failed = True

    for opt, arg in opts:
        if opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-t", "--try-send"):
            try_send = True
        elif opt in ("-s", "--sender-email"):
            sender_email = arg
        elif opt in ("-r", "--reciever-email"):
            reciever_email = arg
        else:
            init_failed = True

    if (is_opts_failed(sender_email, reciever_email, init_failed)):
        print("python ./send_email.py -s <sender-email> -r <reciever-email> -v -t")
        sys.exit()

    # run code after args check

    # Get page data
    html = get_page_urlopen(base_url)
    soup = BeautifulSoup(html, "lxml")
    df = parse_page_with_soup(soup, verbose)

    # Save dataframe as .csv file. Uses current datetime up to seconds for saving path
    save_to_csv(df)

    # Get random data for sending email attempt
    row = get_random_manga(df)
    message = row_to_message(row)

    # Send an email (if try_send variable is `True`)
    send_email(sender_email, reciever_email, message, try_send)


if __name__ == "__main__":
    main(sys.argv[1:])
