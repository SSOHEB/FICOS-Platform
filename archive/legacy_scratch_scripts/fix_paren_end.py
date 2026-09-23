with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('f.write(exec_report)\n"""', 'f.write(exec_report)\n""")')

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed line 640 in build_exp9_full_spec.py to end with \")\"\".")
