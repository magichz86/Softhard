import argparse
import torch
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--plm_eval_mode", action="store_true")
args = parser.parse_args()


# load dataset
from utils import load_data

raw_dataset = load_data("../data/SimpleQuestions/Simple_Question_processed.json")
# Note that if you are running this scripts inside a GPU cluster, there are chances are you are not able to connect to huggingface website directly.
# In this case, we recommend you to run `raw_dataset = load_dataset(...)` on some machine that have internet connections.
# Then use `raw_dataset.save_to_disk(path)` method to save to local path.
# Thirdly upload the saved content into the machine in cluster.
# Then use `load_from_disk` method to load the dataset.

from openprompt.data_utils import InputExample

dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in raw_dataset[split]:
        input_example = InputExample(guid=data["guid"], text_a=data['triple'], text_b=data['answer'], tgt_text=data['question'])
        dataset[split].append(input_example)
# print(dataset['train'][0])

# load plm related things
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
template_text = 'Triple: {"placeholder":"text_a"}. Question: {"special": "<eos>"} {"mask"}.'
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=template_text, using_decoder_past_key_values=False)


# To better understand how does the template wrap the example, we visualize one instance.

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

# Now, the wrapped example is ready to be pass into the tokenizer, hence producing the input for language models.
# You can use the tokenizer to tokenize the input by yourself, but we recommend using our wrapped tokenizer, which is a wrapped tokenizer tailed for InputExample.
# The wrapper has been given if you use our `load_plm` function, otherwise, you should choose the suitable wrapper based on
# the configuration in `openprompt.plms.__init__.py`.
# Note that when t5 is used for classification, we only need to pass <pad> <extra_id_0> <eos> to decoder.
# The loss is calcaluted at <extra_id_0>. Thus passing decoder_max_length=3 saves the space
# from openprompt.plms import T5TokenizerWrapper
#
# wrapped_t5tokenizer = T5TokenizerWrapper(max_seq_length=128, decoder_max_length=128, tokenizer=tokenizer, truncate_method="head")
#
# # You can see what a tokenized example looks like by
# # could set decoder_max_length according to what is shown
# tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
# # print("Example:")
# # print(tokenized_example)
# # print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
# # print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))
#
# # Now it's time to convert the whole dataset into the input format!
# # Simply loop over the dataset to achieve it!
#
# model_inputs = {}
# for split in ['train', 'validation', 'test']:
#     model_inputs[split] = []
#     for sample in dataset[split]:
#         tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
#         model_inputs[split].append(tokenized_example)

# We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")


# ## Now is time to build your prompt model!
# In this section we introduce using prompt to do classification, for other kinds of format, please see
# `generation_tutorial.ipynb`, `probing_tutorial.ipynb`.

print(next(iter(train_dataloader)))
# exit()

from openprompt import PromptForGeneration

use_cuda = True
prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric


# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        print("evaluate step "+str(step))
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence


generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5
    # "bad_words_ids": [[628], [198]],
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(5):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step += 1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        # loss.requires_grad_(True)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step % 500 == 0:
            print("Epoch {}, global_step {} loss: {} lr: {}".format(epoch, global_step, loss.item(), scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

generated_sentence = evaluate(prompt_model, test_dataloader)

with open(f"Generated_sentence_simplequestion.txt", 'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")
