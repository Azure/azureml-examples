import sys
from pathlib import Path

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Start printing Argument List:')
for arg in sys.argv:
    print(f"arg: {arg}")
print('End printing Argument List:')
print("----------------------------------")

input_file_path = Path(sys.argv[1])
print(f"input_file_name: {input_file_path}")
print("----------------------------------")

print(f"input_file_path.exists: {input_file_path.exists()}")
print(f"input_file_path.is_file: {input_file_path.is_file()}")
print(f"input_file_path.is_dir: {input_file_path.is_dir()}")
print(f"input_file_path.is_symlink: {input_file_path.is_symlink()}")
print("----------------------------------")


if input_file_path.is_file():
    print(f"input_file_path.read_text(): {input_file_path.read_text()}")
else:
    print(f"Start listing dir {input_file_path}")
    for x in input_file_path.iterdir():
        print(x)
    print(f"End listing dir {input_file_path}")

    hard_coded_path = input_file_path / "some_file.txt"
    print("Start reading hard_coded_path")
    print(f"{hard_coded_path.read_text()}")
    print("End reading hard_coded_path")

    print(f"Start listing all files in {input_file_path}")
    p = input_file_path.glob('**/*')
    files = [x for x in p if x.is_file()]
    print(f"files: {files}")
    print(f"End listing all files in {input_file_path}")