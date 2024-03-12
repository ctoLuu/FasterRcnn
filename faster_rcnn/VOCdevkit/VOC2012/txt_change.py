import os
import re

path = './label_txt/'
files = []

for file in os.listdir(path):
    if file.endswith(".txt"):
        files.append(path + file)

for file in files:
    with open(file, 'r') as f:
        new_data = re.sub('^0', 'good', f.read(), flags=re.MULTILINE)
    with open(file, 'w') as f:
        f.write(new_data)
    with open(file, 'r') as f:
        new_data = re.sub('^1', 'broke', f.read(), flags=re.MULTILINE)
    with open(file, 'w') as f:
        f.write(new_data)
    with open(file, 'r') as f:
        new_data = re.sub('^2', 'lose', f.read(), flags=re.MULTILINE)
    with open(file, 'w') as f:
        f.write(new_data)
    with open(file, 'r') as f:
        new_data = re.sub('^3', 'uncovered', f.read(), flags=re.MULTILINE)
    with open(file, 'w') as f:
        f.write(new_data)
    with open(file, 'r') as f:
        new_data = re.sub('^4', 'circle', f.read(), flags=re.MULTILINE)
    with open(file, 'w') as f:
        f.write(new_data)


