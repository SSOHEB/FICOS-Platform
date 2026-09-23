with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace exec_report line with chr(10).join
text = text.replace('exec_report = "\\n".join(report_lines)', 'exec_report = chr(10).join(report_lines)')

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated build_exp9_full_spec.py with chr(10).join.")
