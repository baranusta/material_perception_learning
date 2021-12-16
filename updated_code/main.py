from config import *
from train import *
from test import *

def main(is_test):
    if is_test==True:
        test(config)
    else:
        train(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--is_test', default=True, metavar='Boolean',
                        help="Do the training or testing, default is 'False'.")
    
    args = parser.parse_args()
    # main(is_test=True)
    main(is_test=args.is_test)