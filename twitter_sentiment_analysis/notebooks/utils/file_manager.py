from gc import collect
import os
import re
import argparse
from datetime import datetime
from collections import defaultdict

# EXCLUDE = r'|'.join(['\.ipynb$', '\.py$'])
EXCLUDE_FILE_PREFIXES = ('__', '.')
EXCLUDE_FILE_EXT = ('.ipynb', '.py')

timestamp_pattern = r'\d{4}-\d{2}-\d{2}[^\.]+'
timestamp_format = '%Y-%m-%d_%H-%M-%S'

PIPELINE_DATA_DIRS = [
    {
        'twitter_sentiment_analysis/notebooks/pipeline_1/': [
            'config',
            'models/kmeans',
            'models/linear_svc',
            'models/multi_nb',
            'models/vectorizer',
            'models/word_vecs',
            'data/transformed/', 
            'data/embeddings/'
        ]
    },
    {
        'twitter_sentiment_analysis/notebooks/pipeline_2/': [
            'config',
            'models/kmeans',
            'models/linear_svc',
            'models/multi_nb',
            'models/vectorizer',
            'models/word_vecs',
            'data/transformed/', 
            'data/embeddings/',
            'data/cv_scores/linear_svc',
            'data/cv_scores/multi_nb',
            'data/metrics/linear_svc',
            'data/metrics/multi_nb',
            'data/predictions/linear_svc',
            'data/predictions/multi_nb'
        ]
    }
]



def cleanPipelineOutput(pipelines, keep=1, strip_most_recent=True, purge=False, targets=[], verbose=False, automated=True):
    if not pipelines:
        return 
    if not hasattr(pipelines, '__iter__') or isinstance(pipelines, str):
        pipelines = [pipelines]
    
    target_dirs = [PIPELINE_DATA_DIRS[i-1] for i in pipelines if i in range(1, len(PIPELINE_DATA_DIRS)+1)]
    target_dirs = { k: v for d in target_dirs for k, v in d.items() }
    directory_paths = [outer + inner for outer, l in target_dirs.items() for inner in l]

    def parseAndStage(dir_path):
        # Collect all files in the specified directory
        filter_func = lambda x: not x.startswith(EXCLUDE_FILE_PREFIXES) and not x.endswith(EXCLUDE_FILE_EXT)
        filenames = list(filter(filter_func, next(os.walk(dir_path), (None, None, []))[2]))
        staged_paths, removed_files, renamed_files = [], [], []

        if not filenames:
            if verbose:
                print('No files found in directory:', dir_path)
        else:
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

            # Determine which files will be removed or renamed
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
                            if verbose:
                                print('Found path to strip:')
                                print(f'{most_recent_path} -->\n{new_path}')
                            abort = False
                            if os.path.exists(stripped_path):
                                if not automated:
                                    control = input(f'\nNeed to delete existing stripped file:\n{stripped_path}\n\nOK to remove and rename? (y/n) ')
                                else:
                                    control = 'y'
                                if control == 'y':
                                    os.remove(stripped_path)
                                    staged_paths = [x for x in staged_paths if x != stripped_path]
                                    if verbose:
                                        print('\nExisting file successfully removed.')
                                    removed_files.append(stripped_path)
                                else:
                                    abort = True
                            else:
                                control = input(f'\n\nOK to continue? (y/n) ')
                                if control != 'y':
                                    abort = True

                            if abort:
                                if verbose:
                                    print('\nRename aborted.\n')
                                continue

                            os.rename(most_recent_path, new_path)
                            if verbose:
                                print('New file successfully renamed.')
                                print('\n------------------\n')
                            renamed_files.append((most_recent_path, new_path))

                    staged_paths = [x for x in staged_paths if x != stripped_path]

        return staged_paths, removed_files, renamed_files

    collected_paths, removed_files, renamed_files = [], [], []
    for p in directory_paths:
        staged, removed, renamed = parseAndStage(p)
        collected_paths.extend(staged)
        removed_files.extend(removed)
        renamed_files.extend(renamed)
    
    if collected_paths:
        if verbose:
            print('\nPreparing to permenantly delete the following files:\n')
            for x in collected_paths:
                print(x)
                
        control = input('\nOK to continue? (y/n) ') if not automated else 'y'

        if control == 'y':
            for path in collected_paths:
                os.remove(path)
                if verbose:
                    print('Removed file:', path)
                removed_files.append(path)
        elif verbose:
            print('\nAborted file removal.')
    elif verbose:
        print('No files to remove.')

    if verbose:
        print('\n<done>')
    
    return removed_files, renamed_files



cleanPipelineOutput(pipelines=2, strip_most_recent=True, purge=False, verbose=True, automated=True)


