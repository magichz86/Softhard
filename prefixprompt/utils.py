import json


def load_data(data_path):
    print("Loading data...")
    lines = open(data_path, encoding='utf-8').readlines()
    # 分为8:1:1
    max_num = 0
    raw_data = {'train': [], 'validation': [], 'test': []}
    for i, line in enumerate(lines):
        data = json.loads(line)
        data_dump = {}

        #
        s1 = " , ".join(data["seq"])
        data_dump["seq"] = s1

        # h-type , r1.r2.r3 , t-type
        s2 = " , ".join(data["typed_seq"])
        data_dump["typed_seq"] = s2

        data_dump["question"] = data["questions"]
        data_dump["guid"] = i
        if i % 10 < 8:
            raw_data["train"].append(data_dump)
        elif i % 10 == 8:
            raw_data["validation"].append(data_dump)
        else:
            raw_data["test"].append(data_dump)

    #print(raw_data["validation"][0])

    return raw_data


if __name__ == '__main__':
    load_data("../data/SimpleQuestions/SQ.json")