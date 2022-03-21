
# Create an augmented file with roughly 1.000.000 lines
with open('X1.csv') as file:
    lines = file.readlines()
    writes_per_line = int(100000 / len(lines[1:]))
    with open('X1_extended_100000.csv', "w") as output_file:
        output_file.write(lines[0])
        for line in lines[1:]:
            for i in range(0, writes_per_line + 1):
                output_file.write(line)
