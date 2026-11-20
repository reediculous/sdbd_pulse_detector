import os
import zipfile
import numpy as np
import re


class NpzFileIterator:
    def __init__(self, directory, skip=0):
        """Initialize the iterator with an option to skip files."""
        self.directory = directory
        self._files = sorted([file for file in os.listdir(directory) if file.endswith('.npz')])

        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if skip > len(self._files):
            raise ValueError(f"Skip value ({skip}) exceeds the number of available files ({len(self._files)})")

        self._index = skip
        self.skipped = skip

    def __iter__(self):
        return self

    def __next__(self):
        while self._index < len(self._files):
            file = self._files[self._index]
            self._index += 1

            match = re.search(r'(\d+)Ohm_(\d+)V_(\d+)kHz', file)
            if match:
                resistance = int(match.group(1))
                voltage = int(match.group(2))
                frequency = int(match.group(3))
            else:
                print(f"Warning: Skipping file '{file}' - filename does not match expected pattern.")
                continue

            filepath = os.path.join(self.directory, file)
            try:
                data = np.load(filepath)['data']
                data = {
                    't': data[0],
                    'v': data[1],
                    'i': data[2] / 50
                }
                return voltage, frequency, data, filepath
            except zipfile.BadZipfile:
                print(f"Warning: Skipping corrupted file '{file}'")
                continue

        raise StopIteration

    def reset(self, skip=None):
        """Reset the iterator with an optional new skip value."""
        if skip is None:
            skip = self.skipped

        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if skip > len(self._files):
            raise ValueError(f"Skip value ({skip}) exceeds the number of available files ({len(self._files)})")

        self._index = skip
        self.skipped = skip
