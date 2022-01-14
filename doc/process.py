""" Process source files before running Sphinx"""
import re
import os
import shutil
from pygit2 import Repository

repo = 'https://github.com/KaidiXu/auto_LiRPA'
branch = os.environ.get('BRANCH', None) or Repository('.').head.shorthand
repo_file_path = os.path.join(repo, 'tree', branch)

""" Parse README.md into sections which can be reused """
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
print()

""" Load source files from src/ and fix links to GitHub """
for filename in os.listdir('src'):
    print(f'Processing {filename}')
    with open(os.path.join('src', filename)) as file:
        source = file.read()
    source_new = ''
    ptr = 0
    # res = re.findall('\[.*\]\(.*\)', source)
    for m in re.finditer('(\[.*\])(\(.*\))', source):
        assert m.start() >= ptr
        source_new += source[ptr:m.start()]
        ptr = m.start()
        source_new += m.group(1)
        ptr += len(m.group(1))
        link_raw = m.group(2)
        while len(link_raw) >= 2 and link_raw[-2] == ')':
            link_raw = link_raw[:-1]
        link = link_raw[1:-1]
        if link.startswith('https://') or link.startswith('http://') or '.html#' in link:
            print(f'Skip link {link}')
            link_new = link
        else:
            link_new = os.path.join(repo_file_path, 'docs/src', link)
            print(f'Fix link {link} -> {link_new}')
        source_new += f'({link_new})'
        ptr += len(link_raw)
    source_new += source[ptr:]
    with open(filename, 'w') as file:
        file.write(source_new)
    print()