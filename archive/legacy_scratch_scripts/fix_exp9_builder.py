with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace ending of exec_report block
text = text.replace("    f.write(exec_report)\n''')", "    f.write(exec_report)\n\"\"\"")

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed build_exp9_full_spec.py string closing.")
