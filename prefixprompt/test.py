from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor


dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("../data/webnlg_challenge_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("../data/webnlg_challenge_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("../data/webnlg_challenge_2017/")
print("This is the first example of train set:")
print(dataset['train'][4655])