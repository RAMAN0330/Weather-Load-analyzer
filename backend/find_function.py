
import re

with open(r'c:\Users\RamanSharma\OneDrive - GNA-Energy\Desktop\RD\backend\main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'def _compute_dayahead' in line:
        print(f'Found at line {i+1}: {line.strip()}')
