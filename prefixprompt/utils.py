import json


def load_data(data_path):
    print("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data_all = json.load(f)
    # 分为8:1:1
    max_num = 0
    raw_data = {'train': [], 'validation': [], 'test': []}
    for i, data in enumerate(data_all):
        data_dump = {}
        s = "[CLS] "
        # for Simple Question
        s += " [SEP] ".join(data["Triples"])
        data_dump["triple"] = s
        data_dump["answer"] = data["Answer"]
        data_dump["question"] = data["Question"]
        data_dump["guid"] = i
        if i % 10 < 8:
            raw_data["train"].append(data_dump)
        elif i % 10 == 8:
            raw_data["validation"].append(data_dump)
        else:
            raw_data["test"].append(data_dump)

    print(raw_data["validation"][0])

    return raw_data


if __name__ == '__main__':
    load_data("../data/SimpleQuestions/Simple_Question_processed.json")
