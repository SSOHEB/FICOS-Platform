with open('scratch/build_exp9_full_spec.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = """        cases.append({
            "date": dt,
            "vessel": v,
            "horizon": h,
            "decision": dec,"""

new_block = """        cases.append({
            "date": dt,
            "vessel": v,
            "horizon": h,
            "y0": y0,
            "y_true": y_true,
            "decision": dec,"""

text = text.replace(old_block, new_block)

with open('scratch/build_exp9_full_spec.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Added y0 and y_true to cases dictionary in build_exp9_full_spec.py")
