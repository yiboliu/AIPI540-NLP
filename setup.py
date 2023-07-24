from zipfile import ZipFile


def unzip_file(file_name):
    with ZipFile(file_name, 'r') as zf:
        zf.extractall()


if __name__ == "__main__":
    unzip_file('tweet-sentiment-extraction.zip')
