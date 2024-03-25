import subprocess
import shlex

file = open("gc4_corpus_head_urls.txt")
urls = file.readlines()
file.close()

numUrls = 30
for url in urls:
    print(url)
    subprocess.call(shlex.split(f"./convert_file.sh {url}"))
    print(numUrls, " left to download.")
    numUrls = numUrls - 1