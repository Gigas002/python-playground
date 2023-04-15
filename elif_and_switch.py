# requires python 3.10+

# if-elif switch emulation (pre-3.10)
def switch(lang):
    if lang == "JavaScript":
        return "You can become a web developer."
    elif lang == "Python":
        return "You can become a Data Scientist"
    elif lang == "PHP":
        return "You can become a backend developer."
    elif lang == "Solidity":
        return "You can become a Blockchain developer."
    elif lang == "Java":
        return "You can become a mobile app developer"
    else:
        return "The language doesn't matter, what matters is solving problems."


# switch sequence on python 3.10+
def modern_switch(lang):
    match lang:
        case "JavaScript":
            return "You can become a web developer."
        case "Python":
            return "You can become a Data Scientist"
        case "PHP":
            return "You can become a backend developer"
        case "Solidity":
            return "You can become a Blockchain developer"
        case "Java":
            return "You can become a mobile app developer"
        case _:
            return "The language doesn't matter, what matters is solving problems."


def main():
    print(switch("JavaScript"))
    print(switch("PHP"))
    print(switch("Java"))

    print(modern_switch("JavaScript"))
    print(modern_switch("PHP"))
    print(modern_switch("Java"))


main()
