import os
import re
import argparse
from datetime import datetime
from collections import defaultdict


timestamp_pattern = r'\d{4}-\d{2}-\d{2}[^\.]+'
timestamp_format = '%Y-%m-%d_%H-%M-%S'


def cleanDataOutput(keep=1, strip_most_recent=False, targets=[], purge=False, verbose=False):

    target_dirs = { 
        'notebooks/pipeline_1/': ['models/', 'data/transformed/', 'data/embeddings/']
    }

    directory_paths = [outer + inner for outer, l in target_dirs.items() for inner in l]

    def parseAndStage(dir_path):
        # Collect all files in the specified directory
        filenames = next(os.walk(dir_path), (None, None, []))[2]

        if not filenames and verbose:
            print('No files found in directory:', dir_path)

        # Extract timestamps for ordering
        file_dates = defaultdict(list)
        for f_name in filenames:
            if (date := re.findall(timestamp_pattern, f_name)):
                name = re.split(timestamp_pattern, f_name)[0].rstrip('_')
                file_dates[name].append((datetime.strptime(date[0], timestamp_format), f_name))
            else:
                extracted_date = datetime.fromtimestamp(os.path.getmtime(os.path.join(dir_path, f_name)))
                file_dates[f_name.split('.')[0]].append((extracted_date, f_name))
        
        # Filter possible paths based on target groups
        if targets:
            file_dates = { k: v for k, v in file_dates.items() if k in targets }

        # Determine which files will be removed
        staged_paths = []
        for name, dates in file_dates.items():
            if purge:
                staged_paths.extend(map(lambda x: os.path.join(dir_path, x[1]), dates))
            else:
                ordered_dates = sorted(dates, key=lambda x: x[0], reverse=True)
                staged_paths.extend(map(lambda x: os.path.join(dir_path, x[1]), ordered_dates[keep:]))

                most_recent_path = os.path.join(dir_path, ordered_dates[0][1])
                path_components = re.split(timestamp_pattern, ordered_dates[0][1])
                
                if len(path_components) == 2:
                    stripped_path = os.path.join(dir_path, path_components[0].rstrip('_') + path_components[1])
                else:
                    stripped_path = path_components[0]

                
                if strip_most_recent and len(ordered_dates) >= keep:
                    new_path = ''.join([x.rstrip('_') for x in re.split(timestamp_pattern, most_recent_path)])
                    if (most_recent_path != new_path):
                        print('Found path to strip:')
                        print(f'{most_recent_path} -->\n{new_path}')
                        abort = False
                        if os.path.exists(stripped_path):
                            user_input = input(f'\nNeed to delete existing stripped file:\n{stripped_path}\n\nOK to remove and rename? (y/n) ')
                            if user_input == 'y':
                                os.remove(stripped_path)
                                staged_paths = [x for x in staged_paths if x != stripped_path]
                                print('\nExisting file successfully removed.')
                            else:
                                abort = True
                        else:
                            user_input = input(f'\n\nOK to continue? (y/n) ')
                            if user_input != 'y':
                                abort = True

                        if abort:
                            print('\nRename aborted.\n')
                            continue

                        os.rename(most_recent_path, new_path)
                        print('New file successfully renamed.')
                        print('\n------------------\n')

                print(stripped_path)
                staged_paths = [x for x in staged_paths if x != stripped_path]

        return staged_paths


    collected_paths = []

    for p in directory_paths:
        collected_paths.extend(parseAndStage(p))
    
    if (collected_paths):
        print('\nPreparing to permenantly delete the following files:\n')
        for x in collected_paths:
            print(x)
        user_input = input('\nOK to continue? (y/n) ')
        if user_input == 'y':
            print()
            for path in collected_paths:
                os.remove(path)
                print('Removed file:', path)
        else:
            print('\nAborted file removal.')
    else:
        print('No files to remove.')

    print('\n<done>')



cleanDataOutput(strip_most_recent=True, purge=False, verbose=True)







# Initialize parser
parser = argparse.ArgumentParser()

# Add clean models arg
# parser.add_argument("clean_models", help="display a square of a given number",
#                     type=int)

# parser.parse_args()

