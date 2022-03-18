import argparse
import torch
import wandb
import os
import json

from utils import load_data
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from openprompt.utils.metrics import generation_metric
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from openprompt.prompts import MixedTemplate
parser = argparse.ArgumentParser("")
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
args = parser.parse_args()


def main(generation_arguments):
    # load dataset
    raw_dataset = load_data("../data/SimpleQuestions/SQ.json")
    dataset = {}
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for data in raw_dataset[split]:
            input_example = InputExample(guid=data["guid"], text_a=data['seq'], text_b=data['typed_seq'], tgt_text=data['question'])
            dataset[split].append(input_example)
    print("This is the first example of train set:")
    print(dataset['train'][0])

    # load plm related things
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

    # Instantiating the PrefixTuning Template !
    template_text = ' [CLS] {"placeholder":"text_a"} [SEP] {"placeholder":"text_b"} [SEP] {"special": "<eos>"} {"mask"}'
    mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=template_text, num_token=10, using_decoder_past_key_values=False)

    # To better understand how does the template wrap the example, visualize one instance.
    wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
    print("This is the wrapped example of train set 1:")
    print(wrapped_example)

    print("Looping data...")
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
        batch_size=5, shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
        batch_size=5, shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=128,
        batch_size=5, shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head")
    print("Finished looping data...")

    # load the pipeline model PromptForGeneration.
    prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
    if args.use_cuda:
        prompt_model = prompt_model.cuda()

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
    tot_step = len(train_dataloader)*5
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # training and generation.
    global_step = 0
    tot_loss = 0
    log_loss = 0
    for epoch in range(5):
        prompt_model.train()
        for step, inputs in enumerate(train_dataloader):
            global_step += 1
            if args.use_cuda:
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
                print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
                log_loss = tot_loss
        if not os.path.isdir("saved_dict"):
            os.makedirs("saved_dict")
        torch.save(prompt_model.state_dict(), './saved_dict/' + "Epoch" + str(epoch) + "_" + args.model_name_or_path + '.pt')

    #prompt_model.load_state_dict(torch.load('./saved_dict/sq-Epoch4_t5-base.pt'))
    generated_sentence, groundtruth_sentence = evaluate(prompt_model, test_dataloader, generation_arguments)

    with open(f"Generated_sentence_simplequestion.json", 'w') as f:
        for i, s in enumerate(generated_sentence):
            dump = {"generated_sentence": s, "groundtruth_sentence": groundtruth_sentence[i]}
            f.write(json.dumps(dump, ensure_ascii=False) + '\n')


# Define evaluate function
def evaluate(prompt_model, dataloader, generation_arguments):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    print("Evaluating on test data...")
    for step, inputs in enumerate(dataloader):
        if args.use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        #print("evaluate step " + str(step))
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_bleu_4: ", 100*score, flush=True)
    return generated_sentence, groundtruth_sentence


if __name__ == '__main__':

    generation_arguments = {
        "max_length": 128,
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
    main(generation_arguments)
