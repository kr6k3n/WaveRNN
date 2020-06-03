# importing required modules
from zipfile import ZipFile
import os


def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, _ , files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


def compress_filepath(directory, destination=None, silent=True):
    if destination is None:
        destination = directory + ".zip"
    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    if not silent:
        print('Following files will be zipped:')
        for file_name in file_paths:
            print(file_name)

    with ZipFile(destination, 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file)

    if not silent: print('All files zipped successfully!')


def decompress_filepath(file, destination=None, silent=True):
    zip_ref = ZipFile(file, 'r')
    zip_ref.extractall(destination)
