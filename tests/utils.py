import numpy as np


def print_arrays(*arrs):
    np.set_printoptions(linewidth=300)
    np.set_printoptions(threshold=np.inf)
    print()
    arr_strings_lines = [
        np.array2string(
            arr,
            separator=", ",
            formatter={"float_kind": lambda x: f"{x:.3f}"},
        ).split("\n")
        for arr in arrs
    ]

    line_lengths = [len(lines[0]) for lines in arr_strings_lines]

    while True:
        num_empty = 0
        for i, lines in enumerate(arr_strings_lines):

            if len(lines) > 0:
                line = lines.pop(0)
                print(line + "   ", end="")
            else:
                print(" " * line_lengths[i] + "   ", end="")
                num_empty += 1
        print()
        if num_empty >= len(arrs):
            break
