import time
import signal
import threading
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager


### Singleton Logger
class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # call original __new__
        return cls._instance  # return current instance

    def __init__(self):
        if not Logger._initialized:
            self.timing = {}
            Logger._initialized = True

    def add_timing(self, name, time):
        if not name in self.timing:
            self.timing[name] = Data()

        self.timing[name].add_data(time)

    def clear_timing(self, name=None):
        for timing_name, data in self.timing.items():
            if name is None or timing_name == name:
                data.clear()

    def log_timing(self, figsize=(12, 7)):
        if len(self.timing) == 0:
            return

        # pie chart labels
        labels = self.timing.keys()

        # pie chart time splits
        time_split = np.array([data.sum() for data in self.timing.values()])
        total_time = time_split.sum()
        time_split = time_split / total_time

        # on key press code
        def on_key(event):
            if event.key == 'q':
                plt.close()

        # time split pie chart
        plt.figure(figsize=figsize)
        plt.pie(time_split, labels=labels, autopct='%1.1f%%')
        plt.title('Time Responsbility')
        plt.legend(
            labels,
            loc='lower left',
            bbox_to_anchor=(-0.0541666667 * figsize[0], -0.0214285714 * figsize[1]),
        )

        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        # per timing step plot
        for name, data in self.timing.items():
            plt.figure(figsize=figsize)
            plt.margins(x=0)
            plt.xticks(range(0, len(data.data), max(1, int(len(data.data) / 10))))
            plt.title(name)
            plt.xlabel('Log index')
            plt.ylabel('Time (ms)')
            plt.step(range(len(data.data)), data.data)
            plt.axhline(data.average(), color='red')

            plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            plt.show()


class Data:
    def __init__(self):
        self.data = []

    def add_data(self, data):
        self.data.append(data)

    def clear(self):
        self.data.clear()

    def sum(self):
        return sum(self.data)

    def average(self):
        if not self.data:
            return 0.0
        return sum(self.data) / len(self.data)


@contextmanager
def timer(name, timeout_s=180):
    def signal_handler(signum, frame):
        timeout_flag = True
        raise RuntimeError(f'{name} timed out after {timeout_s}s')

    # store start time
    start = time.perf_counter()

    # start signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(timeout_s))
    try:
        yield
    finally:
        # log time
        end = time.perf_counter()
        length = 1000 * (end - start)  # convert to ms
        (Logger()).add_timing(name, round(length, 5))  # log

        # disable signal
        signal.alarm(0)
