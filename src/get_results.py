"""
Methods to get results using Evaluators for Toybox and ImageNet+COCO-12
"""
from Evaluator import IN12Evaluator, ToyboxEvaluator

if __name__ == "__main__":
    exp_name = "Mar-18-2022-01-09"
    evaluator = IN12Evaluator(out_dir_name=exp_name)
    evaluator.calc_acc()
    evaluator.get_class_accuracy()
    # evaluator.get_imgnet_coco_split_giraffe()
    # evaluator.get_acc_by_candidates()
    evaluator.compare_two_models(reverse=True)
    evaluator.compare_two_models(reverse=False)
    evaluator.compare_two_models(reverse=True, same=True)
    evaluator.compare_two_models(reverse=False, same=True)
    
    evaluator = ToyboxEvaluator(out_dir_name=exp_name)
    evaluator.calc_acc()
    evaluator.get_class_accuracy()
    # evaluator.get_acc_by_candidates()
    evaluator.compare_two_models(reverse=True)
    evaluator.compare_two_models(reverse=False)
    evaluator.compare_two_models(reverse=True, same=True)
    evaluator.compare_two_models(reverse=False, same=True)
    
    