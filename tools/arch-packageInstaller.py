# this file is so I can install needed packages system wide on my arch installation
# run file as sudo to avoid needing to put password in at start

import os

with open("requirements.txt") as file:
    
    for line in file.readlines():
        print(line)
