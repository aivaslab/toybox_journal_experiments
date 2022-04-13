"""
Module for project utils
"""
import torch
import argparse
import logging


def calc_accuracy(output, target, topk=(1,)):
	"""
	Computes the accuracy over the k top predictions for the specified values of k
	"""
	with torch.no_grad():
		maxk = max(topk)
		batchSize = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
		correct_top_k = torch.eq(pred, target_reshaped)
		pred_1 = pred[0]
		res = []
		for k in topk:
			correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(torch.mul(correct_k, 100.0 / batchSize))
		return res, pred_1
	
	
def get_parser():
	"""
	Use argparse to get parser
	"""
	parser = argparse.ArgumentParser(description="Parser")
	parser.add_argument("--log-level", "-l", default="warning", type=str)
	parser.add_argument("--config-file", "-conf", default="./default_config.yaml", type=str)
	parser.add_argument("--seed", "-s", default=0, type=int)
	return parser.parse_args()


def tryAssert(fn, arg, msg):
	try:
		assert fn(arg), msg
	except AssertionError:
		logging.error(msg)
		raise
