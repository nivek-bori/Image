import numpy as np
import motmetrics as mm
from util.matching import calculate_iou

class MOTDataFrame:
	def __init__(self, frame, ids, xywhs):
		self.frame = frame
		self.ids = ids
		self.xywhs = np.array(xywhs)

def calculate_cost_mat(gt_xywh, ts_xywh):
	# format ground truth and tracklet xywh arrays into 2D mat
	row_xywh = gt_xywh[:, np.newaxis, :]
	col_xywh = ts_xywh[np.newaxis, :, :]

	# calculate iou
	iou = calculate_iou(row_xywh, col_xywh)

	# calculate cost
	cost = np.fill_like(iou, fill_value=1.0)
	cost += 1 - iou
	return cost

def evaluate(gt_list, ts_list):
	if len(gt_list) != len(ts_list):
		raise ValueError('Length of ground truths array and tracklets must be same array')
	
	acc = mm.MOTAccumulator(auto_id=True)
	
	for gt, ts in zip(gt_list, ts_list):
		frame = gt.frame

		gt_ids = gt.ids
		ts_ids = ts.ids

		cost_mat = calculate_cost_mat(gt.xywhs, ts.xywhs)

		acc.update(
			gt_ids,
			ts_ids,
			cost_mat,
			frameid=frame
		)
		
	mh = mm.metrics.create()
	summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'idsw', 'fp', 'fn'], name='ByteTrack')
	print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metrics))