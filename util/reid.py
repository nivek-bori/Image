import torch


### REID

class Reid():
	def __init__(self, type, max_num, reid_shape, mult=1):
		match type:
			case 'ave':
				self._reid = Ave_Reid(max_num, reid_shape)
			case 'time':
				self._reid = Weight_Reid(max_num, reid_shape, mult=1)
			case 'weight':
				self._reid = Weight_Reid(max_num, reid_shape, mult=mult)
			case _:
				raise ValueError(f'Unknown reid type {type}')
	
	def step_reid(self, *args, **kwargs):
		return self._reid.step_reid(*args, **kwargs)
	
	def get_reid(self):
		return self._reid.get_reid()
	
	def __str__(self, *args, **kwargs):
		return self._reid.__str__(*args, **kwargs)

class Ave_Reid:
	def __init__(self, max_num, reid_shape):
		self.max_num = max_num

		self.ave = torch.zeros(reid_shape)
		self.reids = torch.zeros((max_num, *reid_shape))

		self.head = 0
		self.num = 0
	
	def step_reid(self, reid):
		reid = torch.tensor(reid)

		reid = reid / self.max_num # to save compute, ave is assumed to be ( x / max_num )

		self.ave -= self.reids[self.head] # remove old reid. if no old reid, it is minus by zero
		self.ave += reid # add new reid to average

		self.reids[self.head] = reid # replace/add old reid with new
		self.head = (self.head + 1) % self.max_num # increment pos
		self.num = min(self.max_num, self.num + 1)
	
	def get_reid(self):
		# return average as is
		if self.num == self.max_num:
			return self.ave
		
		# recalibrate average from ( x / max_num ) to ( x / num )
		else:
			return self.ave * (self.max_num / self.num)

	def __str__(self):
		return f'ave: {self.ave}, num: {self.num}'

# weight by time. x weight = inital weight * pow(mult, age)
class Weight_Reid:
	def __init__(self, max_num, reid_shape, mult):
		self.max_num = max_num
		self.mult = mult
		self.curr_weight = 0
		self.max_weight = self.calculate_weight(self.max_num) # maximum weight 
		self.max_single_weight = self.calculate_single_weight(self.max_num) # maximum weight on single value

		self.reids = torch.zeros((max_num, *reid_shape))
		self.sum = torch.zeros(*reid_shape) # weighted sum
		self.step = torch.zeros(*reid_shape) # unweighted step

		self.head = 0
		self.num = 0

	def step_reid(self, reid):
		reid = torch.tensor(reid)

		self.step -= self.reids[self.head] # remove current reid
		self.sum -= self.reids[self.head] * self.max_single_weight # remove currrent weighted reid
		self.step += reid # add new reid

		self.sum *= self.mult # weigh reid
		self.sum += self.step # add all reid

		self.reids[self.head] = reid # replace/add old reid with new
		self.head = (self.head + 1) % self.max_num # increment pos
		self.num = min(self.max_num, self.num + 1)
	
	def get_reid(self):
		# use precalculated max weight or recalculate
		self.curr_weight = self.max_weight if self.num == self.max_num else self.calculate_weight(self.num)
		weighted_reid = self.sum / self.curr_weight # weighted average 

		return weighted_reid
	
	# total weight of num elements
	def calculate_weight(self, num):
		return sum([x * pow(self.mult, num - x) for x in range(1, num + 1)])

	# total weight of element # { num }
	def calculate_single_weight(self, num):
		return sum(pow(self.mult, x) for x in range(0, num))
	
	def __str__(self):
		return f'sum: {self.sum}, step: {self.step}, weighted: {self.curr_weight}, weight_sum: {self.get_reid()}, num: {self.num}, mult: {self.mult}, max: {self.max_weight} & {self.max_single_weight}'


### TESTING

def test_ave_reid():
	reid_cls = Reid('ave', 3, (3, ))
	for _t in range(5):
		# reid_cls.step_reid((1, 1, 1))
		reid_cls.step_reid((2, 3, 2))
		reid_cls.get_reid()
		print(reid_cls)
		

def test_weight_reid():
	reid_cls = Reid('weight', 3, (3, ), mult=1.0)
	for _t in range(5):
		# reid_cls.step_reid((1, 1, 1))
		reid_cls.step_reid((2, 3, 2))
		reid_cls.get_reid()
		print(reid_cls)