
import json
import time 
import os 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Split a file into multiple files")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the file to be split")
    parser.add_argument("--num_files", type=int, required=True, help="Number of files to split into")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split files")
    parser.add_argument("--index_to_file_mapping", type=str, required=True, help="Path to save the index to file mapping")
    return parser.parse_args()



def split(path: str,     # Path to the file to be split
          num_files: int, # Number of files to split into
          output_dir: str # Directory to save the split files
          ):
    """
    Split the file into num_files files and save them in output_dir
    """
    now = time.time()
    with open (path) as f: 
        data = f.readlines()

    print("Time taken to read the file is ", time.time()-now)
    os.makedirs(output_dir, exist_ok=True)
    len_in_each_file = len(data)//num_files + 1
    print(len_in_each_file)
    index_to_file_mapping = {}

    for i in range(num_files):
        # print(i)
        for j in range(len_in_each_file):
            if i*len_in_each_file+j < len(data):
                index_to_file_mapping[int(i*len_in_each_file+j)] = [f"{output_dir}/{i}.jsonl" , j]

    print(len(index_to_file_mapping))
    for key, testue in index_to_file_mapping.items():
        with open(f"{testue[0]}", "a") as f:
            f.write(data[key])
    return index_to_file_mapping

if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    num_files = args.num_files
    output_dir = args.output_dir
    index_to_file_mapping = args.index_to_file_mapping
    try:
        mapping = split(input_file, num_files, output_dir)
        # print("Splitting complete. Mapping:", mapping)
    except (ValueError, FileNotFoundError) as e:
        print(e)
    with open(index_to_file_mapping, "w") as f:
        json.dump(mapping, f)


