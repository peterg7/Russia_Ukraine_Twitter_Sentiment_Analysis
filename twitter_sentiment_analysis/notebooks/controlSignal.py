
import os
import sys
from io import StringIO
from enum import IntEnum

class ControlSignal:
    class ACTIONS(IntEnum):
        ABORT = 1
        WARNING = 2
        INFO = 3
        UNKNOWN = 9
        def __str__(self):
            if self is ControlSignal.ACTIONS.ABORT:
                return "ERROR: Aborting execution."
            if self is ControlSignal.ACTIONS.WARNING:
                return "Warning: Able to continue execution but be aware."
            if self is ControlSignal.ACTIONS.INFO:
                return "INFO: Can be ignored."
            if self is ControlSignal.ACTIONS.UNKNOWN:
                return "UNKNOWN: No further information."

    class FLAGS(IntEnum):
        INVALID_CONFIG_TYPE = 10
        MISSING_AREA = 11
        MISSING_REQUIRED = 12
        INVALID_REQUIRED = 13
        USER_INPUT = 14
        INVALID_LOCATION = 15
        UNKNOWN = 99
        def __str__(self):
            if self is ControlSignal.FLAGS.INVALID_CONFIG_TYPE:
                return "Unreadable config due to invalid type. {message}"
            if self is ControlSignal.FLAGS.MISSING_AREA:
                return "Missing pipeline config area. {message}. Using default area."
            if self is ControlSignal.FLAGS.MISSING_REQUIRED:
                return "Missing a required config value. {message}"
            if self is ControlSignal.FLAGS.INVALID_LOCATION:
                return "Could not access location. {message} Ignoring location."
            if self is ControlSignal.FLAGS.INVALID_REQUIRED:
                return "Could not interpret a required config value. {message}"
            if self is ControlSignal.FLAGS.USER_INPUT:
                return "Received user input to exit. {message}"
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
        return str(self.action) + '\n' + str(self.details).format(message=self.msg)
    
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
        return self.action < other.action
  
    def __gt__(self, other):
        return self.action > other.action
  
    def __le__(self, other):
        if self.action == other.action:
            return self.details <= other.details
        return self.action < other.action
  
    def __ge__(self, other):
        if self.action == other.action:
            return self.details <= other.details
        return self.action > other.action
  
    def __eq__(self, other):
        return self.action == other.action and self.details == other.details

    def __ne__(self, other):
        return self.action != other.action or self.details != other.details

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.action.name, (self.details.name).lower())
        

CONTROL_FLAGS = ControlSignal.FLAGS
CONTROL_ACTIONS = ControlSignal.ACTIONS


def processSignals(signals, destination_map, terminate=False):
    if not signals:
        return
    
    # Prioritize any abort signals
    abort_signal = None
    if isinstance(signals, ControlSignal) and signals.abort:
        abort_signal = signals
    elif isinstance(signals[0], ControlSignal): # Assume iterable
        abort_signal = next((x for x in signals if x.abort), None)

    signals.sort(reverse=True)
    for signal in signals:
        print(signal)

    if abort_signal:

        # Process aborted. Clean up...
        staged_paths = [path for path in destination_map.values() if os.path.isfile(path)]
        if staged_paths:
            print('Cleaning...')
            for path in staged_paths:
                try:
                    print('REMOVING:', path)
                    # os.remove(path)
                except OSError as e:
                    print('Could not delete file. Error:' + str(e))
        print('\n<done>')
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
