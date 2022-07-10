
import os
import sys
from io import StringIO
from enum import IntEnum

import numpy as np

class ControlSignal:
    class ACTIONS(IntEnum):
        ABORT = 1
        WARNING = 2
        INFO = 3
        UNKNOWN = 9
        def __str__(self):
            if self is ControlSignal.ACTIONS.ABORT:
                return "[ERROR] Aborting execution - "
            if self is ControlSignal.ACTIONS.WARNING:
                return "[Warning] Able to continue execution but be aware - "
            if self is ControlSignal.ACTIONS.INFO:
                return "[INFO] "
            if self is ControlSignal.ACTIONS.UNKNOWN:
                return "[UNKNOWN] Unknown issue - "

    class FLAGS(IntEnum):
        ### Config-Parsing ###
        INVALID_CONFIG_TYPE = 10
        MISSING_AREA = 11
        MISSING_REQUIRED = 12
        INVALID_REQUIRED = 13
        INVALID_LOCATION = 14
        MISSING_NECCESSARY = 15
        MISSING_IMPORT_LOC = 16
        MISSING_EXPORT_LOC = 17
        MISSING_OPTIONAL_ARGS = 18

        ### Run-Time Execution ###
        FILE_MANAGEMENT = 30
        IMPORT_EXCEPTION = 31
        USER_INPUT = 32

        UNKNOWN = 99

        def __str__(self):
            if self is ControlSignal.FLAGS.INVALID_CONFIG_TYPE:
                return "Unreadable config due to invalid type. {message}"
            if self is ControlSignal.FLAGS.MISSING_AREA:
                return "Missing pipeline config area. {message}"
            if self is ControlSignal.FLAGS.MISSING_REQUIRED:
                return "Missing a required config value. {message}"
            if self is ControlSignal.FLAGS.INVALID_LOCATION:
                return "Could not access location. {message}"
            if self is ControlSignal.FLAGS.INVALID_REQUIRED:
                return "Could not interpret a required config value. {message}"
            if self is ControlSignal.FLAGS.MISSING_IMPORT_LOC:
                return "Could not find import parameter: {message}. Will not import this object."
            if self is ControlSignal.FLAGS.MISSING_EXPORT_LOC:
                return "Could not find export parameter: {message}. Will not export this object."
            if self is ControlSignal.FLAGS.MISSING_OPTIONAL_ARGS:
                return "Could not find optional parameter: {message}. Will use default argument(s)."
            
            if self is ControlSignal.FLAGS.FILE_MANAGEMENT:
                return "While manipulating the file system: {message}"
            if self is ControlSignal.FLAGS.IMPORT_EXCEPTION:
                return "While importing data: {message}"
            if self is ControlSignal.FLAGS.USER_INPUT:
                return "User-caused. {message}"
            
            if self is ControlSignal.FLAGS.UNKNOWN:
                return "No details available. {message}"

    def __init__(self, flags, *args, **kwargs):
        self.action, self.details, self.msg = None, None, ''
        self.as_tuple = lambda : (self.action, self.details, self.msg)

        if isinstance(flags, tuple) or isinstance(flags, list):
            self.importParams(flags)
        else:
            self[0] = flags

        if args and (empty_indicies := [i for i, x in enumerate(self.as_tuple())if not x]):
            self.importParams(args, empty_indicies[0])
        
        if kwargs:
            if (action := kwargs.get('action')):
                self.action = action
            if (details := kwargs.get('details')):
                self.details = details
            if (msg := kwargs.get('msg')):
                self.msg = msg

        if not self.action:
            self.action = ControlSignal.ACTIONS.UNKNOWN
        if not self.details:
            self.details = ControlSignal.FLAGS.UNKNOWN

        self.abort = self.action == ControlSignal.ACTIONS.ABORT
    
    def importParams(self, items, index=0):
        for i, val in enumerate(items):
            self[i+index] = val
        

    def __str__(self):
        return str(self.action) + str(self.details).format(message=self.msg)
    
    def __getitem__(self, index):
        return self.as_tuple()[index]

    def __setitem__(self, index, val):
        if index == 0:
            self.action = val if isinstance(val, ControlSignal.ACTIONS) else ControlSignal.ACTIONS.UNKNOWN
        elif index == 1:
            self.details = val if isinstance(val, ControlSignal.FLAGS) else ControlSignal.FLAGS.UNKNOWN
        elif index == 2:
            self.msg = val
    

    def __lt__(self, other):
        if isinstance(other, ControlSignal):
            return self.action < other.action
        return self.action < other
  
    def __gt__(self, other):
        if isinstance(other, ControlSignal):
            return self.action > other.action
        return self.action > other
  
    def __le__(self, other):
        if isinstance(other, ControlSignal):
            if self.action == other.action:
                return self.details <= other.details
            return self.action < other.action
        return self.action <= other
        
    def __ge__(self, other):
        if isinstance(other, ControlSignal):
            if self.action == other.action:
                return self.details >= other.details
            return self.action > other.action
        return self.action >= other
  
    def __eq__(self, other):
        if isinstance(other, ControlSignal):
            return self.action == other.action and self.details == other.details
        return self.action == other

    def __ne__(self, other):
        if isinstance(other, ControlSignal):
            if self.action == other.action:
                return self.details != other.details
            return False
        return self.action != other

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.action.name, (self.details.name).lower())
        

CONTROL_FLAGS = ControlSignal.FLAGS
CONTROL_ACTIONS = ControlSignal.ACTIONS


def processSignals(signals, generated_files=[], log_level=None, terminate=False):
    
    if not signals:
        return
    
    if not log_level or not isinstance(log_level, ControlSignal.ACTIONS):
        log_level = ControlSignal.ACTIONS.WARNING

    if isinstance(signals, ControlSignal):
        signals = [signals]
    elif not isinstance(signals[0], ControlSignal):
        return
    
    np_signals = np.array(sorted(signals, reverse=True))
    abort_mask = [s.abort for s in signals]
    abort_signals, std_signals = np_signals[abort_mask], np_signals[np.logical_not(abort_mask)]
    
    filtered_signals = list(filter(lambda x: x <= log_level, std_signals))
    for sig in filtered_signals:
        print(sig)

    if np.any(abort_signals):
        for sig in abort_signals:
            print(sig)

        # Process aborted. Clean up...
        staged_paths = [path for path in generated_files if os.path.isfile(path)]
        if staged_paths:
            print('Cleaning...')
            for path in staged_paths:
                try:
                    print('REMOVING:', path)
                    # os.remove(path)
                except OSError as e:
                    print('Could not delete file. Error:' + str(e))
        print('\nTerminating Process...')
        exit_nb()
    
    if terminate:
        print('\n<done>')
    


class JupyterExit(SystemExit):
    def __init__(self):
        sys.stderr = StringIO()

    def __del__(self):
        sys.stderr.close()
        sys.stderr = sys.__stderr__  # restore from backup


def exit_nb():
    raise JupyterExit
