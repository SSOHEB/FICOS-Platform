with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace inner triple quotes in generated report string
text = text.replace('report = f"""', 'report = f\'\'\'')
text = text.replace('print(report)\n"""', 'print(report)\n\'\'\'')

# Also check if any other inner triple quotes exist inside code(""" ... """)
# We can replace triple quotes inside report string with triple single quotes '''
lines = text.split('\n')
in_code = False
fixed_lines = []

for line in lines:
    fixed_lines.append(line)

new_text = '\n'.join(fixed_lines)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Updated build_exp9_full_spec.py")
