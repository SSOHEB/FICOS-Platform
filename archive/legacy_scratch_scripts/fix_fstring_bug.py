with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace line with f-string error
old_line = 'print(f"                         ((S_t + S_{t+h})/2) * 20d + $8,000 * h * 0.25")'
new_line = 'print("                         ((S_t + S_{t+h})/2) * 20d + $8,000 * h * 0.25")'

text = text.replace(old_line, new_line)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed f-string bug in build_exp9_full_spec.py")
