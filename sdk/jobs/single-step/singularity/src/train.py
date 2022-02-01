import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", required=False, action="store")
    args = parser.parse_args()

    print(f"Learing rate is {args.lr}")
