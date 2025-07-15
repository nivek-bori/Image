import time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

### Singleton Logger
class Logger:
	_instance = None
	_initialized = False

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls) # call original __new__
		return cls._instance # return current instance
	
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
	
	def log_timing(self):
		# timing split pie chart
		labels = self.timing.keys()
		
		time_split = np.array([data.sum() for data in self.timing.values()])
		total_time = time_split.sum()
		time_split = time_split / total_time

		plt.figure(figsize=(10, 6))
		plt.pie(time_split, labels=labels)
		plt.title('Time Responsbility')

		def on_key(_event):
			plt.close()
		plt.gcf().canvas.mpl_connect('key_press_event', on_key)
		plt.show()
		
		# per timing step plot
		for name, data in self.timing.items():
			plt.figure(figsize=(10, 6))
			plt.margins(x=0)
			plt.xticks(range(0, len(data.data), max(1, int(len(data.data) / 10))))
			plt.title(name)
			plt.xlabel('Log index')
			plt.ylabel('Time (ms)')
			plt.step(range(len(data.data)), data.data)
			plt.axhline(data.average(), color='red')

			def on_key(_event):
				plt.close()
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
def timer(name):
	start = time.perf_counter()
	try:
		yield
	finally:
		end = time.perf_counter()
		length = 1000 * (end - start) # convert to ms
		( Logger() ).add_timing(name, round(length, 5)) # log