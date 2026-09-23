with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == 'report = f"""':
        lines[i] = "report = f'''\n"
    elif line.strip() == '============================================================' and i > 1460 and i < 1475:
        # line after 1469
        if lines[i+1].strip() == '"""':
            lines[i+1] = "'''\n"

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed line 1470 in build_exp9_full_spec.py")
