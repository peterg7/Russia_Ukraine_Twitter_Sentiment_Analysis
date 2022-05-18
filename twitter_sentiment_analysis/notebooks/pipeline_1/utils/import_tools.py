
# Import dependencies

# File system manipulation
from shared_imports import (rmfile, mvfile, ospath, rmdir, listdir, 
                                ControlSignal, CONTROL_ACTIONS, CONTROL_FLAGS)

# Importing remote datasets
import opendatasets as od



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