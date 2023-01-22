import zipfile
from pathlib import Path
import argparse

# zip all the whl files in the wheel directory

def zip_whl_files(wheel_dir, zip_file):
    with zipfile.ZipFile(zip_file, 'w') as zip:
        for whl in Path(wheel_dir).glob('*.whl'):
            zip.write(whl, whl.name, compress_type=zipfile.ZIP_DEFLATED)
            
    
def main():
    parser = argparse.ArgumentParser(description='Zip all the whl files in the wheel directory')
    parser.add_argument('--wheel_dir', help='The directory where the whl files are located')
    parser.add_argument('--zip_file', help='The zip file to create')
    args = parser.parse_args()
    zip_whl_files(args.wheel_dir, args.zip_file)

if __name__ == '__main__':
    main()

