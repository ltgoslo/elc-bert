from smart_open import open
from normalize import clean


def preprocess(f):
    prev_line = None
    for line in f:
        line = ' '.join(line.strip().split())
        line = clean(line, minimal=True)

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        if len(line) == 0:
            if prev_line is None or len(prev_line) > 0:
                yield ""
            prev_line = line
            continue

        if not line.endswith(":") and not line.startswith('"') and not line.endswith('"') and not (line.startswith("(") and line.endswith(")")) and not (line.startswith("[") and line.endswith("]")) and not (line.startswith("{") and line.endswith("}")):
            line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


with open("../data/babylm_data/babylm_100M/open_subtitles.train") as f:
    with open("../data/processed/open_subtitles.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")