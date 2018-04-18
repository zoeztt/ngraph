import re

file_name = "build/compile_bench_out.txt"
regex = r'compile time: (.+?)ms'

with open(file_name, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = " ".join(lines)
    compile_time = re.findall(regex, lines)
    print(compile_time)