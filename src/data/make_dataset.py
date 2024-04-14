import zipfile
import argparse
from os.path import join as pathjoin


def extract_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--raw_zip_path', help='Input zip file path')
    parser.add_argument("-o",'--raw_unzipped_path', help='Output unzipped file path')
    args = parser.parse_args()

    if not args.raw_zip_path:
        input_dir = pathjoin("..","..","data","raw","base.zip")
    else: 
        input_dir = args.raw_zip_path

    if not args.raw_unzipped_path:
        output_dir = pathjoin("..","..","data","raw")
    else:
        output_dir = args.raw_unzipped_path

    with zipfile.ZipFile(input_dir,"r") as zip_ref:
        zip_ref.extractall(output_dir)


if __name__ == "__main__":
    extract_dataset()