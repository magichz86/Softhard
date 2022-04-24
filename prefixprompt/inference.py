import json

from nltk.tokenize import word_tokenize

from bert_score import score
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


def cal_bert_score(hypos, refs):
    _, _, b_score = score(hypos, refs, lang='en', verbose=True)
    return b_score.mean()


def cal_rougel(hypos, refs):
    # generated_sens = []
    # ref_sens = []
    # for ref, hypo in zip(refs, hypos):
    #     tokenized_rs = []
    #     ref = ref.split("\n")
    #     for r in ref:
    #         tokenized_rs.append(word_tokenize(r))
    #     hypo = word_tokenize(hypo)
    #     generated_sens.append(hypo)
    #     ref_sens.append(tokenized_rs)
    rouge = Rouge()
    scores = rouge.get_scores(hypos, refs, avg=True)
    return scores["rouge-l"]["f"]


def cal_meteor(hypos, refs):
    m_score = 0
    for line in zip(refs, hypos):
        ref = word_tokenize(line[0])
        hypo = word_tokenize(line[1])
        m_score += meteor_score([ref], hypo)

    return m_score/len(hypos)


if __name__ == '__main__':
    hypos = ["this is an apple", "an apple on this tree"]
    refs = ["an apple on this tree", "an apple on this tree"]
    print(cal_meteor(hypos, refs))

# for ref, hypo in zip(refs, hypos):
#     tokenized_rs = []
#     ref = ref.split("\n")
#     for r in ref:
#         tokenized_rs.append(word_tokenize(r))
#     hypo = word_tokenize(hypo)
#     try:
#         sc = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
#     except ValueError:
#         logger.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
#         sc = 0.0
#     scores.append(sc)
# score = sum(scores) / len(scores)
