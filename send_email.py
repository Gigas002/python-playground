# for details see: https://realpython.com/python-send-email/

import smtplib
import ssl
import getpass
import sys
import getopt

# globals

port = 465

# Works only with gmail sender account

smtp_server = "smtp.gmail.com"

# Message template with subject

message = "\nSubject: Test\n\n"

def is_opts_failed(sender_email, reciever_email, message, init_failed):
    '''
    Check if options were initialized correctly
    '''

    if init_failed:
        return True
    elif not sender_email or not reciever_email or not message:
        return True

# main func

def main(argv):
    opts, args = getopt.getopt(argv, "hs:r:m:", ["sender-email=", "reciever-email=", "message="])
    sender_email = ""
    reciever_email = ""
    send_message = ""
    init_failed = False

    if (len(opts) <= 0):
        init_failed = True

    for opt, arg in opts:
        if opt in ("-s", "--sender-email"):
            sender_email = arg
        elif opt in ("-r", "--reciever-email"):
            reciever_email = arg
        elif opt in ("-m", "--message"):
            send_message = arg
        else:
            init_failed = True

    if (is_opts_failed(sender_email, reciever_email, send_message, init_failed)):
        print("python ./send_email.py -s <sender-email> -r <reciever-email> -m <message>")
        sys.exit()

    global message
    message += send_message

    password = getpass.getpass()

    # create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, reciever_email, message)

if __name__ == "__main__":
    main(sys.argv[1:])
