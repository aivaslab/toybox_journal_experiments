"""
This module validates the configs from the yaml files
"""


def validate_lr_scheduler(yaml_dict, target_component, num_batches):
	"""
	This method makes sure that for CosineAnnealingLR, T_max=num_epochs
	"""
	assert num_batches is not None, "Reuse method by setting num_batches = len(DataLoader)..."
	comp_args = yaml_dict[target_component]['args']
	if comp_args['lr_scheduler'] == 'CosineAnnealingLR':
		comp_args['lr_scheduler_args'] = {}
		num_epochs_decay = comp_args['num_epochs'] - 2
		comp_args['lr_scheduler_args']['T_max'] = num_epochs_decay * num_batches
		# print(comp_args['lr_scheduler_args']['T_max'])
	elif comp_args['lr_scheduler'] == "StepLR":
		comp_args['lr_scheduler_args']['step_size'] *= num_batches
		comp_args['lr_scheduler_args']['step_size'] += 1

	return yaml_dict


def validate_toybox_views(yaml_dict, target_component):
	"""
	This method that training component on Toybox has views specified.
	"""
	comp_args = yaml_dict[target_component]['args']
	if 'views' not in comp_args:
		comp_args['views'] = ['rxminus', 'rxplus', 'ryminus', 'ryplus', 'rzminus', 'rzplus']
	return yaml_dict

def get_model_name(module, name, *args, **kwargs):
	return getattr(module, name)(*args, **kwargs)


def get_optimizer_name(module, name, *args, **kwargs):
	return getattr(module, name)(*args, **kwargs)


def get_dataset(module, name, *args, **kwargs):
	return getattr(module, name)(*args, **kwargs)


def get_scheduler_name(module, name, *args, **kwargs):
	return getattr(module, name)(*args, **kwargs)
