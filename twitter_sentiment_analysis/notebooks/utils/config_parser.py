
from collections import defaultdict
from operator import itemgetter
import re
import os
import sys
import json
import errno
from datetime import datetime
import fsspec

# from ControlSignal import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS
from shared_imports import ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS, np

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


def _flattenDicts(d):
    if isinstance(d, (list, tuple)):
        flat_dict = {}
        for x in d:
            if isinstance(x, dict):
                flat_dict.update(x)
        return flat_dict

    elif not isinstance(d, dict):
        return {}
    
    return d


def mergeDicts(a, b, no_merge=None, path=None):
    if path is None: 
        path = []
    if no_merge is None:
        no_merge = set([])
    if new_no_merge := b.get('no_merge', a.get('no_merge')):
        no_merge |= set(new_no_merge)
    
    for key in filter(lambda x: x not in no_merge, b):
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergeDicts(a=a[key], b=b[key], no_merge=no_merge, path=(path + [str(key)]))
        else:
            a[key] = b[key]
    return a, no_merge


def mergeConfigs(configs, init_metadata={}, nested_params=None, excluded_params=None):
    as_list = lambda x: [] if not x else [x] if (not hasattr(x, '__iter__') or isinstance(x, (str, dict))) else x
    configs = as_list(configs)
    nested = as_list(nested_params)
    excluded = as_list(excluded_params)

    drop_meta = lambda conf_dict: { k: v for k, v in conf_dict.items() if k != 'metadata' }

    config_data = {}
    config_metadata = {**{"no_merge": set([])}, **init_metadata}
    for conf in configs:
        if isinstance(conf, str):
            with open(conf, 'r') as f:
                data = json.load(f)
        elif isinstance(conf, dict):
            data = conf
        else:
            continue
        
        if (nested_keys := np.intersect1d(list(data.keys()), nested)).size > 0:
            data = _flattenDicts(itemgetter(*nested_keys)(data))
        
        if (included_keys := np.setdiff1d(list(data.keys()), excluded)).size > 0:
            data = { k: v for k, v in data.items() if k in included_keys }
        
        
        if metadata := data.get('metadata'):
            _, no_merge = mergeDicts(config_metadata, metadata)
            config_metadata["no_merge"] |= no_merge

        _, no_merge = mergeDicts(config_data, drop_meta(data), no_merge=config_metadata.get('no_merge'))
        config_metadata["no_merge"] |= no_merge
        
    return config_data, config_metadata


def buildConfig(dflt_configs, usr_configs='', extract_config={}, transform_config={}, 
                        model_config={}, evaluate_config={}, load_config={}, **kwargs):
    signals = []
    pipeline_params = {}

    stage_configs = (extract_config, transform_config, model_config, evaluate_config, load_config)
    if not dflt_configs:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, 
                                        "Must provide default configuration."))
        return signals, None
    
    if isinstance(dflt_configs, str) or hasattr(dflt_configs, '__iter__'):
        dflt_config_data, metadata = mergeConfigs(dflt_configs, **kwargs)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_CONFIG_TYPE, 
                                            "Can not interpret default configuration."))
        return signals, None
    
    if not usr_configs:
        if not any(stage_configs):
            signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS, 
                                            "No user or stage configs. Executing with the default configuration."))
            return signals, { k: v for k, v in dflt_config_data.items() if k != 'metadata' }
        usr_config_data = {}
    elif isinstance(usr_configs, str) or hasattr(usr_configs, '__iter__'):
        usr_config_data, _ = mergeConfigs(usr_configs, init_metadata=metadata, **kwargs)
    else:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_CONFIG_TYPE, 
                                            "Can not interpret provided user configuration."))
        return signals, None

    # Combine user-provided configs
    agg_config_data = defaultdict(dict, {k: v for d in stage_configs for k, v in d.items()})
    agg_config_data.update(usr_config_data)

    # Establish valid config areas
    for k in dflt_config_data.keys():
        if not k in agg_config_data:
            if k in metadata['areas']['required']:
                signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_AREA, f'Could not find {k}'))
                return signals, pipeline_params
            else:
                signals.append(ControlSignal(CONTROL_ACTIONS.INFO , CONTROL_FLAGS.MISSING_AREA, f'Could not find {k}'))
                pipeline_params[k] = dflt_config_data[k]
        else:
            pipeline_params[k] = agg_config_data[k]


    # Check for missing required keys
    agg_dflt_keys = [k for values in dflt_config_data.values() for k in values.keys()]
    agg_user_keys = [k for values in agg_config_data.values() for k in values.keys()]
    missing_params = set(agg_dflt_keys) - set(agg_user_keys)
    missing_required = missing_params.intersection(set(metadata['required_params']))
    if missing_required:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, f"Cound not find the following fields: {' | '.join(missing_required)}"))
        return signals, pipeline_params

    curr_timestamp = re.sub(':', '-', datetime.now().isoformat('_', timespec='seconds'))

    for area, dflt_values in dflt_config_data.items():
        for key, dflt_val in dflt_values.items():
            
            if key in metadata['locations']:
                if (found_path := agg_config_data[area].get(key)):
                    if isinstance(found_path, dict):
                        valid_locations = {}
                        for k, v in found_path.items():
                            if v:
                                formatted_path, timestamp = formatPath(v, agg_config_data, curr_timestamp)
                                valid_path, path_info, path_protocol = validatePath(formatted_path)
                                if valid_path:
                                    valid_locations[k] = formatted_path
                                    valid_locations[f'{k}_protocol'] = path_protocol
                                elif k in metadata['required_params']:
                                    signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}]'))
                                    return signals, pipeline_params
                                else:
                                    signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}].\nAble to substitute with default value.'))
                                    valid_locations[k] = formatted_path
                                    valid_locations[f'{k}_protocol'] = path_protocol
                        pipeline_params[area][key] = valid_locations
                    else:
                        formatted_path, timestamp = formatPath(found_path, agg_config_data, curr_timestamp)
                        valid_path, path_info, path_protocol = validatePath(formatted_path)
                        if valid_path:
                            pipeline_params[area][key] = formatted_path
                            pipeline_params[area][f'{key}_protocol'] = path_protocol
                        elif key in metadata['required_params']:
                            signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}]'))
                            return signals, pipeline_params
                        else:
                            signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}].\nAble to substitute with default value.'))
                            pipeline_params[area][key] = formatted_path
                            pipeline_params[area][f'{key}_protocol'] = path_protocol
                else:
                    if isinstance(dflt_val, dict):
                        valid_locations = {}
                        for k, v in dflt_val.items():
                            if v:
                                formatted_path, timestamp = formatPath(v, agg_config_data, curr_timestamp)
                                valid_path, path_info, path_protocol = validatePath(formatted_path)
                                if valid_path:
                                    valid_locations[k] = formatted_path
                                    valid_locations[f'{k}_protocol'] = path_protocol
                                elif k in metadata['required_params']:
                                    signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}]'))
                                    return signals, pipeline_params
                                else:
                                    signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_LOCATION, f'Failed to validate path: [{formatted_path}].\nAble to substitute with default value.'))
                                    valid_locations[k] = formatted_path
                                    valid_locations[f'{k}_protocol'] = path_protocol
                        pipeline_params[area][key] = valid_locations
                            
                    elif area == 'EXTRACT':
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_IMPORT_LOC, f'{area} - {key}'))
                    elif area == 'LOAD':
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_EXPORT_LOC, f'{area} - {key}'))
                    else:
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.UNKNOWN, f"Occurred while searching for [{area} - {key}]"))

            elif key in metadata['required_params']:
                if (required_param := agg_config_data[area].get(key)):
                        pipeline_params[area][key] = required_param
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.MISSING_REQUIRED, f"Missing required field: [{area} - {key}]"))
                    return signals, pipeline_params

            elif key == 'sentiment_map' or key == 'sentiment_vals':
                if (found_map := agg_config_data[area].get(key)):
                    if key == 'sentiment_vals':
                        found_map = found_map.get('value_mapping')
                        cluster_map = found_map.get('cluster_mapping')
                    else:
                        cluster_map = []

                    converted_map = {}
                    for k, v in found_map.items():
                        try:
                            converted_map[int(k)] = v
                        except Exception as e:
                            converted_map = dflt_val
                            signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.INVALID_CONFIG_TYPE, "Sentiment map must have integer keys. Could not convert provided value."))
                            continue
                    missing_vals = set(metadata['necessary_sentiment_values']) - set(converted_map.values())
                    dflt_replacements = { k: v for k, v in dflt_val.items() if v in missing_vals }
                    converted_map.update(dflt_replacements)
                    
                    if cluster_map:
                        pipeline_params[area][key]['value_mapping'] = converted_map
                        if invalid_vals := set(range(-1, 2)) ^ set(cluster_map):
                            cluster_map.extend(list(invalid_vals))
                        pipeline_params[area][key]['cluster_mapping'] = cluster_map
                        
                    else:
                        pipeline_params[area][key] = converted_map
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS, f'{area} - {key}'))
                    if key == 'sentiment_vals':
                        converted_vals = {
                            'value_mapping': { int(k): v for k, v in dflt_val['value_mapping'].items() },
                            'cluster_mapping': dflt_val['cluster_mapping']
                        }
                    else:
                        converted_vals = { int(k): v for k, v in dflt_val.items() }
                    pipeline_params[area][key] = converted_vals

            elif key in pipeline_params[area]:
                # Some sort of validation...?
                continue
            
            else:
                signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.MISSING_OPTIONAL_ARGS, f'{area} - {key}'))
                pipeline_params[area][key] = dflt_val
    
    pipeline_params['metadata'] = metadata
    return signals, pipeline_params
