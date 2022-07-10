
# Import dependencies

# File system manipulation
import shutil
from joblib import load

import pandas as pd
from control_signal import  ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS
from os import remove as rmfile
from os import rename as mvfile
from os import path as ospath
from os import walk as oswalk
from os import rmdir, listdir

# Importing remote datasets
import opendatasets as od


def importData(import_loc, import_protocol, local_dest=None, expected_type=None, required=True, signals=[], **kwargs):
    failed_action = CONTROL_ACTIONS.ABORT if required else CONTROL_ACTIONS.WARNING
    if import_loc == local_dest:
        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.IMPORT_EXCEPTION,
                        f'Import source and destination are equal. Nothing to download.'))
    else:
        if local_dest:
            # Separate directory from filename
            dest_dir = ospath.dirname(local_dest)
            dest_filename = ospath.basename(local_dest)

            # Collect all files in the destination directory
            existing_dest_files = next(oswalk(dest_dir), (None, None, []))[2]

            # Check if new data needs to be downloaded
            if dest_filename not in existing_dest_files or kwargs.get('overwrite_data'):
                # Remove old data file if present
                if ospath.exists(local_dest):
                    try:
                        rmfile(local_dest)
                    except OSError as e:
                        signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.FILE_MANAGEMENT,
                                                    (f'Could not delete existing raw data file: {local_dest}.' +
                                                    f' Received error {repr(e)}')))
            if import_protocol == 'file':
                try:
                    shutil.copy(import_loc, local_dest)
                except shutil.Error as e:
                    signals.append(ControlSignal(failed_action, CONTROL_FLAGS.FILE_MANAGEMENT,
                                                f'Could not copy local file. Received error {repr(e)}'))
                    if required:
                        return signals, None
            
            elif import_protocol in ['http', 'https']:
                if 'kaggle.com' in import_loc:
                    _, kaggle_dest = importKaggleData(import_loc, local_dest, signals)
                    if not kaggle_dest:
                        return signals, None
                    local_dest = kaggle_dest
                else:
                    signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.IMPORT_EXCEPTION,
                                                f"Unsupported {import_protocol} import location. Received error {repr(e)}. Will still attempt to import."))
                    local_dest = import_loc
            else:
                signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.IMPORT_EXCEPTION,
                                        f"Unrecognized import protocol: {import_protocol}. Will still attempt to import."))
                local_dest = import_loc
        
        else:
            if expected_type and isinstance(import_loc, expected_type):
                return signals, import_loc
            signals.append(ControlSignal(CONTROL_ACTIONS.INFO, CONTROL_FLAGS.FILE_MANAGEMENT,
                                            f'No local destination specified for import of {import_loc}. Will attempt to import directly from the import path.'))
            local_dest = import_loc
    
    if isinstance(local_dest, str):
        if local_dest.endswith('.joblib'):
            import_func = load
        elif local_dest.endswith('.csv'):
            import_func = pd.read_csv
        elif local_dest.endswith('.json'):
            import_func = pd.read_json
        elif import_protocol in ['http', 'https']:
            import_func = pd.read_html
        else:
            signals.append(ControlSignal(failed_action, CONTROL_FLAGS.IMPORT_EXCEPTION,
                                            (f"Unsupported data storage format for import. Attempted to extract data" + 
                                            f"from {local_dest}")))
            if required:
                return signals, None
            import_func = pd.read_table
    
    # Import data
    try:
        import_obj = import_func(local_dest)
    except Exception as e:
        signals.append(ControlSignal(failed_action, CONTROL_FLAGS.FILE_MANAGEMENT,
                                    f'Failed to import data from {local_dest}. Received error {repr(e)}'))
        if required:
            return signals, None
        import_obj = None
    
    if expected_type and not isinstance(import_obj, expected_type):
        signals.append(ControlSignal(failed_action, CONTROL_FLAGS.IMPORT_EXCEPTION,
                                    f'The imported object - {import_obj} did not match the expected type - {expected_type}.'))
        return signals, None
    
    return signals, import_obj


def importKaggleData(url, dest_path, signals=[]):
    
    dest_dir = ospath.dirname(dest_path)

    # Extract imported dataset's filename
    dataset_id = od.utils.kaggle_direct.get_kaggle_dataset_id(url) # in form 'username/dataset_name'
    dataset_name = dataset_id.split('/')[1]
    tmp_dataset_dir = ospath.join(dest_dir, dataset_name)

    # Check if temporary import directory exists (from previous download) and remove it
    if ospath.isdir(tmp_dataset_dir):
        prev_imports = [ospath.join(tmp_dataset_dir, f) for f in listdir(tmp_dataset_dir) if f.endswith('.csv')]
        for file in prev_imports:
            try:
                rmfile(file)
            except OSError as e:
                signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.FILE_MANAGEMENT, 
                                                (f'Could not delete existing raw data file: {file}\n' +
                                                f'Received error {repr(e)}')))
        try:
            rmdir(tmp_dataset_dir)
        except OSError as e:
            signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.FILE_MANAGEMENT, 
                                            (f'Could not remove existing import directory: {tmp_dataset_dir}\n' +
                                            f'Received error {repr(e)}')))
    
    # Download dataset
    od.download(url, data_dir=dest_dir)

    # Check for proper import
    imported_filename = next((f for f in listdir(tmp_dataset_dir) if f.endswith('.csv')), '')
    if not imported_filename:
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.IMPORT_EXCEPTION, 
                                        (f'Failed importing data. File was either moved or not downloaded.' +
                                        f'Expected csv file in {tmp_dataset_dir}')))
        return signals, None

    # Move and rename file
    tmp_dataset_dest = ospath.join(tmp_dataset_dir, imported_filename)
    mvfile(tmp_dataset_dest, dest_path)

    # Remove temporary directory created when downloaded
    if listdir(tmp_dataset_dir):
        signals.append(ControlSignal(CONTROL_ACTIONS.ABORT, CONTROL_FLAGS.FILE_MANAGEMENT, 
                                    (f'Temporary directory {tmp_dataset_dir} is not empty after moving imported file.' +
                                    f'Exiting for safety.')))
        return signals, None
    try:
        rmdir(tmp_dataset_dir)
    except OSError as e:
        signals.append(ControlSignal(CONTROL_ACTIONS.WARNING, CONTROL_FLAGS.FILE_MANAGEMENT, 
                                    f'Could not delete import directory {tmp_dataset_dir}\nReceived error: {repr(e)}'))

    return signals, dest_path