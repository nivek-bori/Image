import torch


### REID

# factory reid
class Reid():
	def __init__(self, type, max_num, reid_shape, mult=1):
		self.type = type
		match self.type:
			case 'ave':
				self._reid = Ave_Reid(max_num, reid_shape)
			case 'time':
				self._reid = Time_Reid(max_num, reid_shape, mult=mult)
			case 'weight':
				self._reid = Weight_Reid(max_num, reid_shape)
			case _:
				raise ValueError(f'Unknown reid type {type}. Valid types are ave, time, weight')
	
	def step_reid(self, reid, conf=None):
		match self.type:
			case 'ave':
				return self._reid.step_reid(reid)
			case 'time':
				return self._reid.step_reid(reid)
			case 'weight':
				return self._reid.step_reid(reid, conf=conf)
	
	def get_reid(self):
		return self._reid.get_reid()
	
	def __str__(self):
		return self._reid.__str__()

class Ave_Reid:
	def __init__(self, max_num, reid_shape):
		self.max_num = max_num

		self.ave = torch.zeros(reid_shape)
		self.reids = torch.zeros((max_num, *reid_shape))

		self.head = 0
		self.num = 0
	
	def step_reid(self, reid):
		if isinstance(reid, torch.Tensor):
			reid = reid.detach().clone()
		else: 
			reid = torch.tensor(reid, dtype=torch.float32)

		reid = reid / self.max_num # to save compute, ave is assumed to be ( x / max_num )

		# remove current reid and add new reid
		self.ave -= self.reids[self.head] # subtracts zero if no head reid
		self.ave += reid

		# update states
		self.reids[self.head] = reid # replace/add current reid with new

		self.head = (self.head + 1) % self.max_num # increment pos
		self.num = min(self.max_num, self.num + 1)
	
	def get_reid(self):
		if self.num == self.max_num or self.num == 0: # return average as is
			return self.ave
		# recalibrate average from ( x / max_num ) to ( x / num )
		return self.ave * (self.max_num / self.num)

	def __str__(self):
		return f'output: {self.get_reid()}, ave: {self.ave}, num: {self.num}'

# weight linearly by time. mult weights older reids more
class Time_Reid:
	def __init__(self, max_num, reid_shape, mult):
		self.max_num = max_num
		self.mult = mult
		self.max_single_weight = self.calculate_single_weight(self.max_num) # maximum weight on single value
		self.max_weight = self.calculate_weight(self.max_num) # maximum weight 

		self.reids = torch.zeros((max_num, *reid_shape))
		self.sum = torch.zeros(*reid_shape) # weighted sum
		self.step = torch.zeros(*reid_shape) # unweighted step

		self.sum_weight = 1
		self.head = 0
		self.num = 0

	def step_reid(self, reid):
		reid = torch.tensor(reid, dtype=torch.float32)

		# remove current reid and add new reid to step
		self.step -= self.reids[self.head]
		self.step += reid

		# remove weighted current reid and weigh and step
		self.sum -= self.reids[self.head] * self.max_single_weight # remove currrent weighted reid
		self.sum *= self.mult # weigh reid
		self.sum += self.step # add all reid

		# update states
		self.reids[self.head] = reid # replace/add old reid with new

		self.head = (self.head + 1) % self.max_num # increment pos
		self.num = min(self.max_num, self.num + 1)
	
	def get_reid(self):
		# use precalculated max weight or recalculate
		self.sum_weight = self.max_weight if self.num == self.max_num else self.calculate_weight(self.num)
		
		# weighted average 
		return self.sum / self.sum_weight
	
	# total weight of num elements
	def calculate_weight(self, num):
		return sum([x * pow(self.mult, num - x) for x in range(1, num + 1)])

	# total weight of element # { num }
	def calculate_single_weight(self, num):
		return sum(pow(self.mult, x) for x in range(0, num))
	
	def __str__(self):
		return f'output: {self.get_reid()}, sum: {self.sum}, step: {self.step}, weight sum: {self.sum_weight}, max: {self.max_weight} & {self.max_single_weight}'

class Weight_Reid:
	def __init__(self, max_num, reid_shape):
		self.max_num = max_num

		self.ave = torch.zeros(reid_shape)
		self.reids = torch.zeros((max_num, *reid_shape))
		self.confs = torch.zeros(max_num, )
		self.sum_conf = 0

		self.head = 0
		self.num = 0
	
	def step_reid(self, reid, conf):
		reid = torch.tensor(reid, dtype=torch.float32)

		reid *= conf # weigh reid by conf

		# remove current reid and add new reid
		self.ave -= self.reids[self.head]
		self.ave += reid

		# remove current conf and add new conf
		self.sum_conf -= self.confs[self.head]
		self.sum_conf += conf

		# update states
		self.reids[self.head] = reid
		self.confs[self.head] = conf

		self.head = (self.head + 1) % self.max_num
		self.num = min(self.max_num, self.num + 1)

	def get_reid(self):
		if self.sum_conf == 0:
			return self.ave
		return self.ave / self.sum_conf # recalibrate average

	def __str__(self):
		return f'output: {self.get_reid()}, ave: {self.ave}, sum_conf: {self.sum_conf}'

### TESTING

def test_ave_reid():
	reid_cls = Reid('ave', 3, (3, ))
	for _t in range(5):
		# reid_cls.step_reid((1, 1, 1))
		reid_cls.step_reid((2, 3, 2))
		reid_cls.get_reid()
		print(reid_cls)
		

def test_time_reid():
	reid_cls = Reid('time', 3, (3, ), mult=1.0)
	for _t in range(5):
		# reid_cls.step_reid((1, 1, 1))
		reid_cls.step_reid((2, 3, 2))
		reid_cls.get_reid()
		print(reid_cls)

def test_weight_reid():
	reid_cls = Reid('weight', 3, (3, ))
	for _t in range(5):
		# reid_cls.step_reid((1, 1, 1), 0.1)
		reid_cls.step_reid((2, 3, 2), 0.5)
		reid_cls.get_reid()
		print(reid_cls)