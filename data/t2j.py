import json
import os
import io
import fnmatch
from json import load, dump
import sys
# from . import BASE_DIR
BASE_DIR = "."

def calculate_distance(customer1, customer2):
    '''gavrptw.uitls.calculate_distance(customer1, customer2)'''
    # print(customer1, customer2)
    return ((customer1['x'] - customer2['x'])**2 + \
        (customer1['y'] - customer2['y'])**2)**0.5

def make_dirs_for_file(path):
    '''gavrptw.uitls.make_dirs_for_file(path)'''
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

def text2json(fn="", customize=False):
    
    text_data_dir = os.path.join(BASE_DIR, 'text_customize' if customize else 'text')
    json_data_dir = os.path.join(BASE_DIR, 'json_customize' if customize else 'json')
    
    if fn!="":
        print(f"fn={fn}")
        fn = os.path.join(text_data_dir, fn)
        print(f"fn = {fn}")

    for text_file in map(lambda text_filename: os.path.join(text_data_dir, text_filename), \
        fnmatch.filter(os.listdir(text_data_dir), '*.txt')):

        if fn != "" and text_file != f"{fn}.txt":
            continue

        json_data = {}
        with io.open(text_file, 'rt', newline='') as file_object:
            locations = []
            for line_count, line in enumerate(file_object, start=1):
                if line_count in [2, 3, 4, 6, 7, 8, 9]:
                    pass
                elif line_count == 1:
                    # <Instance name>
                    json_data['name'] = line.strip()
                elif line_count == 5:
                    # <Maximum vehicle number>, <Vehicle capacity>
                    values = line.strip().split()
                    json_data['number'] = int(values[0])
                    json_data['capacity'] = int(values[1])                
                else:
                    # <Custom number>, <X coordinate>, <Y coordinate>,
                    # ... <Demand>, <Ready time>, <Due date>, <Service time>
                    values = line.strip().split()
                    locations.append({                    
                        'x':        int(values[1]),
                        'y':        int(values[2]),
                        'demand':   int(values[3]),
                        'start':    int(values[4]),
                        'end':      int(values[5]),
                        'service':  int(values[6]),
                    })
        # print(json_data)
        json_data['locs'] = locations
        json_data['dist'] = [[calculate_distance(a, b) for a in locations] for b in locations]
        json_file_name = f"{json_data['name']}.json"
        json_file = os.path.join(json_data_dir, json_file_name)
        print(f'Write to file: {json_file}')
        make_dirs_for_file(path=json_file)
        with io.open(json_file, 'wt', newline='') as file_object:
            dump(json_data, file_object, sort_keys=False, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    # text2json()
    # Print total number of arguments
    print ('Total number of arguments:', format(len(sys.argv)))

    # Print all arguments
    print ('Argument List:', str(sys.argv))

    fn = ""

    if len(sys.argv) > 1:
        fn = sys.argv[1]
    
    text2json(fn=fn, customize=False)
