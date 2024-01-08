from ophyd import Signal
import numpy as np

class LVSignal(Signal):
    def __init__(self, name, value, **kwargs):
        super().__init__(name=name, value=value)
        self.bounds = kwargs.get('bounds')
        self.delta = kwargs.get('delta')
        
    def get(self):
        print(f"Tell LabView to get {self.name}: {self._readback}")
        return self._readback
        ...

    def put(self, value, **kwargs):
        print(f"Tell LabView to put {self.name}: {value}")
        super().put(value, **kwargs)
        ...
