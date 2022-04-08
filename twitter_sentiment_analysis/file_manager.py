import os
import re
import argparse
from datetime import datetime
from collections import defaultdict


timestamp_pattern = r'\d{4}-\d{2}-\d{2}[^\.]+'
timestamp_format = '%Y-%m-%d_%H-%M'


def cleanDataOutput(keep=1, strip_most_recent=False, targets=[], purge=False):

    directory_paths = ['./models', './data/transformed', './data/embeddings']

    def parseAndStage(dir_path):
        # Collect all files in the specified directory
        filenames = next(os.walk(dir_path), (None, None, []))[2]

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
                staged_paths.extend(map(lambda x: x[1], dates))
            else:
                ordered_dates = sorted(dates, key=lambda x: x[0], reverse=True)

                staged_paths.extend(map(lambda x: x[1], ordered_dates[:(keep-1)]))
                if strip_most_recent and len(ordered_dates) <= keep:
                    most_recent_path = os.path.join(dir_path, ordered_dates[-1][1])
                    new_path = ''.join([x.rstrip('_') for x in re.split(timestamp_pattern, most_recent_path)])
                    if (most_recent_path != new_path):
                        user_input = input(f'renaming {most_recent_path} --> {new_path}.\nOK to continue? (y/n) ')
                        if user_input == 'y':
                            os.rename(most_recent_path, new_path)
                            print('file successfully renamed.')
                        else:
                            print('rename aborted.')

        return staged_paths


    collected_paths = []
    for p in directory_paths:
        collected_paths.extend(parseAndStage(p))
    
    if (collected_paths):
        print('Preparing to permenantly delete the following files:')
        for x in collected_paths:
            print(x)
        user_input = input('OK to continue? (y/n) ')
        if user_input == 'y':
            for path in collected_paths:
                os.remove(path)
                print('removed file:', path)
        else:
            print('aborted file removal.')
    else:
        print('No files to remove.')

    print('\n<done>')



cleanDataOutput(strip_most_recent=True)







# Initialize parser
parser = argparse.ArgumentParser()

# Add clean models arg
# parser.add_argument("clean_models", help="display a square of a given number",
#                     type=int)

# parser.parse_args()

