mkdir -p ../data/processed

python3 aochildes.py
python3 bnc_spoken.py
python3 cbt.py
python3 children_stories.py
python3 gutenberg.py
python3 open_subtitles.py
python3 qed.py
python3 simple_wikipedia.py
python3 switchboard.py
python3 wikipedia.py

cat ../data/processed/aochildes.txt ../data/processed/bnc_spoken.txt ../data/processed/cbt.txt ../data/processed/children_stories.txt ../data/processed/gutenberg.txt ../data/processed/open_subtitles.txt ../data/processed/qed.txt ../data/processed/simple_wikipedia.txt ../data/processed/switchboard.txt ../data/processed/wikipedia.txt > ../data/processed/all.txt

python3 segment.py