import re

with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace common non-ascii characters with ascii equivalents
replacements = {
    '→': '->',
    '—': '-',
    '–': '-',
    '≈': '~=',
    '≤': '<=',
    '≥': '>=',
    '±': '+/-',
    '•': '*',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '…': '...'
}

for orig, r in replacements.items():
    text = text.replace(orig, r)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Cleaned build_exp9_full_spec.py successfully.")
