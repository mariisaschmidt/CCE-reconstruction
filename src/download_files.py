import subprocess
import shlex

# Open the file containing the urls
file = open("../data/gc4_corpus_head_urls.txt")
urls = file.readlines()
file.close()

# Download the files
numUrls = 30
for url in urls:
    print(url)
    subprocess.call(shlex.split(f"./convert_file.sh {url}"))
    print(numUrls, " left to download.")
    numUrls = numUrls - 1