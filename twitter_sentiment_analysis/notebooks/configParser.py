
import re
import os
import sys
import json
import requests
import errno
from datetime import datetime
from controlSignal import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS


ERROR_INVALID_NAME = 123

def acceptedPathname(pathname: str) -> bool:
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        _, pathname = os.path.splitdrive(pathname)

        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        if not os.path.isdir(root_dirname):
            return False

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError as exc:
        return False
    else:
        return True

def creatablePathname(pathname: str) -> bool:
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

def validPathname(pathname: str) -> bool:
    try:
        return acceptedPathname(pathname) and \
            (os.path.exists(pathname) or creatablePathname(pathname))
    except OSError:
        return False

# Helper function to add current timestamps to file paths
def formatPath(path, config_data, timestamp=None):
    if not timestamp:
        timestamp = re.sub(':', '-', datetime.now().isoformat('_', timespec='seconds'))
    
    if isinstance(path, str):
        if '{timestamp}' in path:
            return path.format(timestamp=timestamp), timestamp
        elif '{version}' in path:
            version = config_data['EXTRACT'].get('data_import_version')
            if re.search('\..*$', path.split('/')[-1]):
                if version:
                    return path.format(version=f'_{version}'), version
                else:
                    return path.format(version=''), version
            elif 'http' in path:
                return path.format(version=version), version
            return path.format(version=version)
    return path, timestamp
        

def validateLocation(loc):
    if 'http' in loc: # Assume to be url
        try:
            response = requests.get(loc)
            return True, "url"
        except requests.ConnectionError as e:
            return False, e
    elif not validPathname(loc):
        return False, f"Could not find local path: {loc}."
    return True, "file"


def validateConfig(dflt_config, config={}, **kwargs):
    signals = []

    # Import default config
    with open(dflt_config, 'r') as f:
        raw_data = json.load(f)
        dflt_config_data = { k: v for k, v in raw_data.items() if k != 'metadata' }
        config_metadata = raw_data['metadata']

    # Import user config
    if isinstance(config, str):
        with open(config, 'r') as f:
            config_data = json.load(f)
    elif isinstance(config, dict):
        config_data = config
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_CONFIG_TYPE, f'Found type {type(config)}. Must be string path or dict'))
        config_data = dflt_config_data

    # Inject kwargs if any
    agg_config_data = config_data | kwargs # kwargs will overwrite config_data on overlap

    pipeline_params = {}
    pipeline_areas = config_metadata['areas']

    for k in dflt_config_data.keys():
        if not k in agg_config_data:
            if k in pipeline_areas['required']:
                signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_AREA, f'Could not find {k}'))
                return pipeline_params, signals
            else:
                signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.MISSING_AREA, f'Could not find {k}'))
                pipeline_params[k] = dflt_config_data[k]
        else:
            pipeline_params[k] = agg_config_data[k]


    # Check for missing required keys
    agg_dflt_keys = [k for values in dflt_config_data.values() for k in values.keys()]
    agg_user_keys = [k for values in agg_config_data.values() for k in values.keys()]
    missing_params = set(agg_dflt_keys) - set(agg_user_keys)
    missing_required = missing_params.intersection(set(config_metadata['required_params']))
    if missing_required:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, f"Cound not find the following fields: {' | '.join(missing_required)}"))
        return pipeline_params, signals

    curr_timestamp = re.sub(':', '-', datetime.now().isoformat('_', timespec='seconds'))

    for area, dflt_values in dflt_config_data.items():
        for key, dflt_val in dflt_values.items():
            
            if key in config_metadata['locations']:
                if (found_path := agg_config_data[area].get(key)):
                    valid_loc = validateLocation(found_path)
                    print(found_path, valid_loc)
                    if valid_loc[0]:
                        pipeline_params[area][key] = formatPath(found_path, agg_config_data, curr_timestamp)[0]
                    elif key in config_metadata['required_params']:
                        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Attempted to validate: {found_path} using {valid_loc[1]} substitute'))
                        return pipeline_params, signals
                    else:
                        signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f"Replacing {found_path} with default value."))
                        pipeline_params[area][key] = formatPath(dflt_val, agg_config_data, curr_timestamp)[0]

            elif key in config_metadata['required_params']:
                if (found_mapping := agg_config_data[area].get(key)):
                    if key == 'column_mappings':
                        found_values = [v for v in config_metadata['necessary_data_mappings'] if v in found_mapping.values()]
                        if not all(found_values):
                            missing_values = set(config_metadata['necessary_data_mappings']) - set(found_values)
                            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_REQUIRED, f"Missing required mapping value(s): {list(missing_values)}"))
                            return pipeline_params, signals
                        for k, v in found_mapping.items():
                            if v in config_metadata['required_params'] and not k:
                                signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_REQUIRED, f"Empty key [{k}] for required mapping value [{v}]"))
                                return pipeline_params, signals
                        pipeline_params[area][key] = found_mapping
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, f"Missing required field: {key}"))
                    return pipeline_params, signals

            elif key == 'sentiment_map':
                if (found_map := agg_config_data[area].get(key)):
                    converted_map = {}
                    for k, v in found_map.items():
                        try:
                            converted_map[int(k)] = v
                        except Exception as e:
                            converted_map[int(dflt_val[k])] = v
                    missing_vals = set(config_metadata['necessary_sentiment_values']) - set(converted_map.values())
                    dflt_replacements = { k: v for k, v in dflt_val.items() if v in missing_vals }
                    converted_map.update(dflt_replacements)
                        
                    pipeline_params[area][key] = converted_map

            elif key in pipeline_params[area]:
                # Some sort of validation...?
                continue
            
            else:
                pipeline_params[area][key] = dflt_val

    
    return pipeline_params, signals