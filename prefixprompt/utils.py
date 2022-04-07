import json


def load_data_pq(data_path):
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
        # elif i % 10 == 8:
        #     raw_data["validation"].append(data_dump)
        elif data["hop_num"] == 3:
            raw_data["test"].append(data_dump)

    print(raw_data["test"][0])

    return raw_data


def load_data_cwq1(data_path):
    print("Loading data...")
    fp = open("../data/ComplexWebQuestions/cwq_test_input.json", 'w', encoding='utf-8')
    with open(data_path, 'r+', encoding='utf-8') as f:
        lines = json.load(f)
    rel_dict_lines = open("../data/ComplexWebQuestions/dict/relationname_dict.json", 'r+', encoding='utf-8').readlines()
    entans_dict_lines = open("../data/ComplexWebQuestions/dict/entityname_answer_dict.json", 'r+',
                             encoding='utf-8').readlines()
    ent_dict_lines = open("../data/ComplexWebQuestions/dict/entityname_dict.json", 'r+', encoding='utf-8').readlines()


    rel_dict = {}
    for r in rel_dict_lines:
        data_r = json.loads(r)
        rel_dict[data_r["freebase_id"]] = data_r["name"]

    ent_dict = {}
    ent_type_dict = {}
    for e in ent_dict_lines:
        data_e = json.loads(e)
        ent_dict[data_e["freebase_id"]] = data_e["name"]
        ent_type_dict[data_e["freebase_id"]] = data_e["notable_type"]

    entans_dict = {}
    entans_type_dict = {}
    for en in entans_dict_lines:
        data_en = json.loads(en)
        entans_dict[data_en["name"]] = data_en["freebase_id"]
        entans_type_dict[data_en["name"]] = data_en["notable_type"]


    # 分为8:1:1
    raw_data = {'train': [], 'validation': [], 'test': []}
    for i, line in enumerate(lines):
        data = line
        data_dump = {}

        triples = data["triples"]
        seq = []
        typed_seq = []

        unknown_ent_num = 1
        a = data["answers"][0]['answer']
        if a not in entans_dict.keys():
            continue
        else:
            ans_name = a
            ans_type = entans_type_dict[a]

        trackent = {}
        for tri in triples:
            temp_tri_seq = []
            temp_tri_typeseq = []

            if tri == ["#ANSWER1", "LIMIT", "1"]:
                continue
            if "ORDERBY" in tri[1]:
                continue
            for j, t in enumerate(tri):
                if t[0] == "?":
                    if t not in trackent.keys():
                        temp_tri_seq.append("Entity" + str(unknown_ent_num))
                        temp_tri_typeseq.append("Entity")
                        trackent[t] = "Entity" + str(unknown_ent_num)
                        unknown_ent_num += 1
                    else:
                        temp_tri_seq.append(trackent[t])
                        temp_tri_typeseq.append("Entity")
                elif t == "#ANSWER1":
                    temp_tri_seq.append(ans_name)
                    temp_tri_typeseq.append(ans_type)
                elif t in ent_dict.keys():
                    temp_tri_seq.append(ent_dict[t])
                    temp_tri_typeseq.append(ent_type_dict[t])
                elif t in rel_dict.keys():
                    temp_tri_seq.append(t)
                    temp_tri_typeseq.append(".".join(rel_dict[t]))
                elif (t[:2] == "18" or t[:2] == "19" or t[:2] == "20" or t[:2] == "17" or t[:2] == "15" or t[:2] == "16" or t[:2] == "14") and (len(t) == 10 or len(t) == 4 or len(t) == 7):
                    temp_tri_seq.append(t)
                    temp_tri_typeseq.append("Date")
                elif (t[:3] == "\"18" or t[:3] == "\"19" or t[:3] == "\"20" or t[:3] == "\"17" or t[:3] == "\"16" or t[:3] == "\"15") and (len(t) == 12 or len(t) == 6 or len(t) == 9):
                    temp_tri_seq.append(t)
                    temp_tri_typeseq.append("Date")
                elif t == "<" or t == ">":
                    temp_tri_seq.append(t)
                    temp_tri_typeseq.append("Mathematical Symbols")
                else:
                    temp_tri_seq.append(t)
                    else_s = tri[j - 1].split(".")[-1].capitalize().replace("_", " ")
                    temp_tri_typeseq.append(else_s)

            temp_tri_seq[1] = temp_tri_seq[1].split(".")[-1]
            #print(temp_tri_seq[1])
            seq.append(" , ".join(temp_tri_seq))#这里调整格式！！！！！！！！！！！！！！！！！！！！！！！
            typed_seq.append(" , ".join(temp_tri_typeseq))

        data_dump["guid"] = i
        s1 = " # ".join(seq)#这里调整格式！！！！！！！！！！！！！！！！！！！！！！！
        data_dump["seq"] = s1

        # h-type , r1.r2.r3 , t-type
        s2 = " # ".join(typed_seq)
        data_dump["typed_seq"] = s2
        # #
        data_dump["question"] = data["question"]

        #print(data_dump)
        if i % 10 < 8:
            raw_data["train"].append(data_dump)
        elif i % 10 == 8:
            raw_data["validation"].append(data_dump)
        else:
            raw_data["test"].append(data_dump)
        fp.write(json.dumps(data_dump, ensure_ascii=False)+"\n")

    print(raw_data["train"][0])
    return raw_data


def load_data(datapath):
    print("Loading data...")
    lines = open(datapath, 'r+', encoding='utf-8').readlines()
    generated_text_lines = open("../data/ComplexWebQuestions/Generated_text_cwq2.json", 'r+',encoding='utf-8').readlines()
    gen_txt = []
    for g in generated_text_lines:
        data_g = json.loads(g)
        gen_txt.append(data_g)

    raw_data = {'train': [], 'validation': [], 'test': []}
    for i, line in enumerate(lines):
        #print(line)
        data = json.loads(line)
        data_dump = data
        flag = 0
        for d in gen_txt:
            if d["guid"] == data["guid"]:
                data_dump["seq"] = d["generated_sentence"]
                flag = 1
                break
        if flag == 0:
            print(error)

        if data["guid"] % 10 < 8:
            raw_data["train"].append(data_dump)
        elif data["guid"] % 10 == 8:
            raw_data["validation"].append(data_dump)
        else:
            raw_data["test"].append(data_dump)

    print(raw_data["train"][0])

    return raw_data




if __name__ == '__main__':
    load_data_cwq1("../data/ComplexWebQuestions/CWQ.json")
