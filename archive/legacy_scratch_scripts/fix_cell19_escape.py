with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace "\n" with "\\n" in line 633
text = text.replace('exec_report = "\n".join(report_lines)', 'exec_report = "\\n".join(report_lines)')

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed Cell 19 newline escaping in build_exp9_full_spec.py")
