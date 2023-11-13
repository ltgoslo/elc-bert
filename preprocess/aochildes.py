from smart_open import open
from normalize import clean


def preprocess(f):
    prev_line = None
    for line in f:
        line = line.strip()

        line = line[0].upper() + line[1:]
        line = clean(line)
        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


with open("../data/babylm_data/babylm_100M/aochildes.train") as f:
    with open("../data/processed/aochildes.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")