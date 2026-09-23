with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's inspect text around line 547 to 645
lines = text.split('\n')
print("Lines 545 to 645:")
for idx in range(545, min(650, len(lines))):
    print(f"{idx+1}: {lines[idx]}")
