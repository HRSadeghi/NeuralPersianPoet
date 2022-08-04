import nltk.translate.gleu_score as gleu
import numpy as np

def get_bleu_scores(gen, Y_out, blue_max_n_gram = 2):
    max_len = max([len(x) for x in Y_out])
    BLEUs = []
    for i in range(1, max_len, 1):
        BLEU_temp = []
        for j in range(0,max_len - i, i):
            Y_out_i = [x[j:i+j] for j,x in enumerate(Y_out) if len(x) - j >= i]
            gen_i = [x[j:i+j] for j,x in enumerate(gen) if len(x) - j >= i]

            if len(Y_out_i) > 0: 
                references = [[' \t '.join(x).split()] for x in Y_out_i]
                hypotheses = [' \t '.join(x).split() for x in gen_i]
                score_i = gleu.corpus_gleu(references, hypotheses, min_len=1, max_len=blue_max_n_gram)

                BLEU_temp.append(score_i)
        BLEUs.append(np.average(BLEU_temp))

    references = [[' \t '.join(x).split()] for x in Y_out]
    hypotheses = [' \t '.join(x).split() for x in gen]
    overall_bleu_score = gleu.corpus_gleu(references, hypotheses, min_len=1, max_len=blue_max_n_gram)

    return BLEUs, overall_bleu_score



def find_type_of_poem(poem):
    if len(poem) >= 4:
        if poem[0][-1] == poem[1][-1] and\
           poem[2][-1] == poem[3][-1]:
           return 1

        if poem[0][-1] == poem[1][-1] and\
           poem[1][-1] == poem[3][-1]:
           return 2

        if poem[1][-1] == poem[3][-1]:
           return 3
        
        return 4


def rhyme_(poem, separate = False):
    accuracy = 0
    if len(poem) >= 4 and len(poem) %2 == 0:
        type_ = find_type_of_poem(poem[:4])
        if type_ == 1:
            accuracy = 2
            for i in range(0, len(poem[4:]), 2):
                if poem[4:][i][-1] == poem[4:][i+1][-1]:
                    accuracy += 1
            if separate:
                return 2*accuracy/len(poem)     
            return [2*accuracy, len(poem)]

        if type_ == 2:
            accuracy = 2
            for i in range(0, len(poem[4:]), 2):
                if poem[0][-1] == poem[4:][i+1][-1]:
                    accuracy += 1
            if separate:
                return 2*accuracy/len(poem)     
            return [2*accuracy, len(poem)]

        if type_ == 3:
            accuracy = 2
            for i in range(0, len(poem[4:]), 2):
                if poem[1][-1] == poem[4:][i+1][-1]:
                    accuracy += 1
            if separate:
                return 2*accuracy/len(poem)     
            return [2*accuracy, len(poem)]

        accuracy = 0
        for i in range(0, len(poem), 2):
            if poem[i][-1] == poem[i+1][-1]:
                accuracy += 1
        if separate:
            return 2*accuracy/len(poem)     
        return [2*accuracy, len(poem)]
    else:
        if separate:
            return 0    
        return [0, 0]


def find_rhyme_accuracy(context, gen, separate = False):
    acc = 0
    count = 0
    for i in range(len(context)):
        t1, t2 = rhyme_(context[i] + gen[i], False)
        acc += t1 - len(context[i])//2
        count += t2 - len(context[i])//2
    return acc/count
    

