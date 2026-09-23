with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the inner report = f""" with report = f''' and its matching triple quote with '''
old_block = 'report = f"""'
new_block = "report = f'''"

content = content.replace(old_block, new_block)

# Replace the end of report string before print(report)
old_end = 'Effect   : PASS/FAIL/NOT TESTABLE standard applied cleanly.\n"""\nprint(report)'
new_end = "Effect   : PASS/FAIL/NOT TESTABLE standard applied cleanly.\n'''\nprint(report)"

content = content.replace(old_end, new_end)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced triple quotes in report block.")
