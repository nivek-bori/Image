import time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

# singleton logger
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
		# timing split data
		timing_responsibility = {}
		timing_total = 0.0

		for name, data in self.timing.items():
			# timing split data processing
			timing_sum = data.sum()
			timing_responsibility[name] = timing_sum
			timing_total += timing_sum

			# per timing step plot
			plt.figure(figsize=(10, 6))
			plt.margins(x=0)
			plt.tight_layout()
			plt.xticks(range(0, len(data.data), int(len(data.data) / 10)))
			plt.title(name)
			plt.xlabel('Log index')
			plt.ylabel('Time (ms)')
			plt.step(range(len(data.data)), data.data)
			plt.axhline(data.average(), color='red')

		# timing split pie chart
		labels = timing_responsibility.keys()
		split = [time / timing_total for time in timing_responsibility.values()]
		plt.figure(figsize=(15, 9))
		plt.pie(split, labels=labels)
		plt.title('Time Responsbility')

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