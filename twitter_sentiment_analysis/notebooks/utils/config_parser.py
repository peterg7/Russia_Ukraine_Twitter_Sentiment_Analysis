
import re
import os
import sys
import json
import errno
from datetime import datetime
import fsspec

# from ControlSignal import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS
from shared_imports import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS

DFLT_CONFIG_PATH = '../default_config.json'
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
            version = config_data['EXTRACT'].get('import_version', '')
            if re.search('\..*$', path.split('/')[-1]):
                if version:
                    return path.format(version=f'_v{version}'), timestamp
                else:
                    return path.format(version=''), timestamp
            elif 'http' in path:
                return path.format(version=version), timestamp
            return path.format(version=version)
    return path, timestamp
        

SUPPORTED_IMPORT_PROTOCOLS = set(['http', 'https', 'ftp', 'ftps', 's3', 's3a', 'github', 'file'])
def validatePath(path, **kwargs):
    protocol = fsspec.utils.get_protocol(path)
    if protocol not in SUPPORTED_IMPORT_PROTOCOLS:
        return False, repr(NotImplementedError(f'Unsupported import path protocol: {protocol}')), protocol
    
    if protocol == 'file':
        if os.path.isfile(path):
            file_stats = os.stat(path)
            stats_dict = { k[2:]: getattr(file_stats, k) for k in dir(file_stats) if k.startswith('st_') }
            stats_dict.update({ 'name': path, 'type': protocol })
            return True, stats_dict, protocol
        if validPathname(path):
            return True, { 'name': path, 'type': protocol }, protocol
        
        return False, repr(FileNotFoundError(f'Could not find local file: {path}')), protocol

    of, info = None, None
    try:
        # Build file system object
        of = fsspec.open(path, mode='rb')#, kwargs=kwargs)
        # Attempt to open connection and read destination info
        info = of.fs.info(path)
    except Exception as e:
        return False, repr(e), protocol
    finally:
        if of:
            of.close()
    
    return True, info, protocol



def validateConfig(config=None, extract_config={}, transform_config={}, model_config={}, load_config={}):
    signals = []
    pipeline_params = {}

    if not config and (not extract_config and not transform_config):
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, "Must provide a config file or both extract & transform configs."))
        return signals, pipeline_params

    # Import default config
    with open(DFLT_CONFIG_PATH, 'r') as f:
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
        config_data = extract_config | transform_config

    # Combine configs
    agg_config_data = { **config_data, **extract_config, **transform_config, **model_config, **load_config }

    # Establish valid config areas
    for k in dflt_config_data.keys():
        if not k in agg_config_data:
            if k in config_metadata['areas']['required']:
                signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_AREA, f'Could not find {k}'))
                return signals, pipeline_params
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
        return signals, pipeline_params

    curr_timestamp = re.sub(':', '-', datetime.now().isoformat('_', timespec='seconds'))

    for area, dflt_values in dflt_config_data.items():
        for key, dflt_val in dflt_values.items():
            
            if key in config_metadata['locations']:
                if (found_path := agg_config_data[area].get(key)):
                    formatted_path, timestamp = formatPath(found_path, agg_config_data, curr_timestamp)
                    valid_path, path_info, path_protocol = validatePath(formatted_path)
                    if valid_path:
                        pipeline_params[area][key] = formatted_path
                        pipeline_params[area][f'{key}_protocol'] = path_protocol
                    elif key in config_metadata['required_params']:
                        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}]'))
                        return signals, pipeline_params
                    else:
                        signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}].\nAble to substitute with default value.'))
                        pipeline_params[area][key] = formatted_path
                        pipeline_params[area][f'{key}_protocol'] = path_protocol
                else:
                    if area == 'EXTRACT':
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_IMPORT_LOC, f'{area} - {key}'))
                    elif area == 'LOAD':
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_EXPORT_LOC, f'{area} - {key}'))
                    else:
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.UNKNOWN, f"Occurred while searching for [{area} - {key}]"))

            elif key in config_metadata['required_params']:
                if (required_param := agg_config_data[area].get(key)):
                        pipeline_params[area][key] = required_param
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, f"Missing required field: [{area} - {key}]"))
                    return signals, pipeline_params

            elif key == 'sentiment_map':
                if (found_map := agg_config_data[area].get(key)):
                    converted_map = {}
                    for k, v in found_map.items():
                        try:
                            converted_map[int(k)] = v
                        except Exception as e:
                            converted_map = dflt_val
                            signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_CONFIG_TYPE, "Sentiment map must have integer keys. Could not convert provided value."))
                            continue
                    missing_vals = set(config_metadata['necessary_sentiment_values']) - set(converted_map.values())
                    dflt_replacements = { k: v for k, v in dflt_val.items() if v in missing_vals }
                    converted_map.update(dflt_replacements)
                        
                    pipeline_params[area][key] = converted_map
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS, f'{area} - {key}'))
                    pipeline_params[area][key] = { int(k): v for k, v in dflt_val.items() }

            elif key in pipeline_params[area]:
                # Some sort of validation...?
                continue
            
            else:
                signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS, f'{area} - {key}'))
                pipeline_params[area][key] = dflt_val

    return signals, pipeline_params