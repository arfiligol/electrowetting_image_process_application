import ttkbootstrap as ttkb
from tkinter import IntVar

class IntScale(ttkb.Scale):
    def __init__(self, master = None, **kw): # **kw 是傳入的其他指名參數
        # Super Init
        super().__init__(master, **kw)

        # Make sure there is an IntVar, if the user doesn't provide one, create one.
        if 'variable' not in kw:
            self.variable = IntVar()
        else:
            self.variable = kw['variable']

        # Using customized command to pack the origin command (if exist)
        original_command = kw.get("command", lambda value: None)
        self.config(command = lambda value: self._update_value_and_call_original(value, original_command)) # Overwrite the original command
    
    def _update_value_and_call_original(self, value, original_command):
        # Set the value to change in integer
        int_value = int(round(float(value)))
        self.variable.set(int_value)

        # call the original command (if exist)
        original_command(int_value)