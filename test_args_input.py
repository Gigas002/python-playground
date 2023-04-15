import sys
import getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
