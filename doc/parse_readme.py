import re
import os

heading = ''
copied = {}
print('Parsing markdown sections from README:')
with open('../README.md') as file:
    for line in file.readlines():
        if line.startswith('##'):
            heading = line[2:].strip()
        else:
            if not heading in copied:
                copied[heading] = ''
            copied[heading] += line
if not os.path.exists('sections'):
    os.makedirs('sections')
for key in copied:
    if key == '':
        continue
    filename = re.sub(r"[?+\'\"]", '', key.lower())
    filename = re.sub(r" ", '-', filename) + '.md'
    print(filename)
    with open(os.path.join('sections', filename), 'w') as file:
        file.write(f'## {key}\n')
        file.write(copied[key])