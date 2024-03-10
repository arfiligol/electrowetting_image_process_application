import ttkbootstrap as ttkb
from tkinter import IntVar

class IntScale(ttkb.Scale):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        # 确保variable是一个IntVar，如果没有提供，就创建一个
        if 'variable' not in kw:
            self.variable = IntVar()
        else:
            self.variable = kw['variable']

        # 使用自定义的command包装原有的command（如果有的话）
        original_command = kw.get('command', lambda value: None)
        self.config(command=lambda value: self._update_value_and_call_original(value, original_command))

    def _update_value_and_call_original(self, value, original_command):
        # 将值修正为整数
        int_value = int(round(float(value)))
        self.variable.set(int_value)
        
        # 调用原始的command（如果存在）
        original_command(int_value)

# 使用示例
if __name__ == "__main__":
    import tkinter as tk
    
    def slider_updated(value):
        print(f"Slider value: {value}")

    root = tk.Tk()
    int_var = IntVar()
    slider = IntScale(root, from_=0, to=100, length = 200, variable=int_var, command=slider_updated)
    slider.pack()

    root.mainloop()
