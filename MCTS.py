"""
python 3.9.12
tensorflow 2.6.0
keras 2.6.0
gensim 4.1.2
"""

""" built-in packages """
import sys
import re
import pdb
import time
import copy
import random
import pickle
import pathlib
import sqlite3
from math import log, sqrt

""" external packages """
import numpy as np
import jieba

""" 分配GPU (要同時跑兩個時會用到)"""
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth =True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

""" NN packages """
from gensim.models import KeyedVectors
from keras.models import load_model

""" attention layer """
from AttentionWithContext.attention_with_context import AttentionWithContext
from keras_layer_normalization import LayerNormalization

""" Global variables """
MCTS_TIME = 5      # second, default 30
MAX_SENTENCE = 10   # max length of template
MAX_BACKUP = 5
MATCH_TABLE = [["33", "11", "11", "11", "11", "11", "-1", "11", "11", "11", "13", "-1", "-1", "-1", "-1", "11"],
               ["--", "11", "21", "21", "12", "33", "21", "11", "23", "11", "11", "21", "-1", "21", "21", "12"],
               ["--", "--", "21", "21", "32", "32", "12", "11", "33", "11", "11", "11", "-1", "-1", "11", "-1"],
               ["--", "--", "--", "11", "-1", "12", "-1", "11", "12", "11", "11", "31", "-1", "-1", "11", "12"],
               ["--", "--", "--", "--", "11", "11", "21", "21", "21", "11", "21", "21", "-1", "-1", "-1", "11"],
               ["--", "--", "--", "--", "--", "33", "-1", "21", "33", "11", "21", "-1", "-1", "-1", "-1", "11"],
               ["--", "--", "--", "--", "--", "--", "-1", "-1", "13", "-1", "-1", "33", "-1", "-1", "11", "-1"],
               ["--", "--", "--", "--", "--", "--", "--", "11", "13", "11", "11", "13", "11", "-1", "12", "13"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "13", "-1", "-1", "21", "-1", "21", "21", "-1"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "-1", "13", "-1", "31", "-1", "22", "11"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "11", "12", "-1", "-1", "12", "12"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "-1", "-1", "11", "-1", "-1"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "-1", "11", "12", "-1"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "-1", "-1", "12"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "33", "22"],
               ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "11"]]   #16x16
MATCH_TABLE = np.array(MATCH_TABLE, dtype='=U2') # unicode string with 2 digit length
i_lower_tri = np.tril_indices(16, k=-1)          # lower triangle indices with diagonal offset -1(below)
MATCH_TABLE[i_lower_tri] = MATCH_TABLE.T[i_lower_tri]
DATABASE = "ConceptNet"
# DATABASE = "ConceptNet_expand_synonyms"
CURRENT_PATH = pathlib.Path(__file__).parent.resolve()

""" NN global """
MAX_WORD_NUM = 10       # max length of words in a sentence
MAX_SENTENCE_NUM = 5    # max length of sentences in a paragraph

""" Template """
mode = np.zeros((MAX_SENTENCE,), dtype=int)-1           # 0:start concept is search node, 1:start concept isn't search node(end node)
                                                        # 0,1:compound relation, 2:single relation
backup_mode = np.zeros((MAX_SENTENCE,), dtype=int)-1    # -1是因為要初始化為-1
c2_reselect_flag = c3_reselect_flag = False
compound_pos = [["" for x in range(2)] for y in range(MAX_SENTENCE)]                # 如果compound relation的其中一個search position為both(Start或End都可以)，在不同的template裡使用情況也各有不同，所以根據不同情況，自行給定search_position(1:Start, 2:End)
single_pos = ["" for x in range(MAX_SENTENCE)]
SurfaceText = ["" for x in range(MAX_SENTENCE)]
template = [["" for x in range(2)] for y in range(MAX_SENTENCE)]
backup_template = [["" for x in range(MAX_BACKUP)] for y in range(MAX_SENTENCE)]    # use backup_template when original one can't be used
                                                                                    # 不管第一組備用是否為single，第二組都要從backup_template[][2]開始放
assoc_flag = [["" for x in range(3)] for y in range(MAX_SENTENCE)]
sentiment = [["" for x in range(3)] for y in range(MAX_SENTENCE)]

""" ConceptNet DB """
CN_start = list()
CN_end = list()
CN_relation = list()
CN_surfacetext = list()
CN_start_concept = list()
""" Cilin DB """
cilin_data = list()
cilin_coding_sub1 = list()
cilin_coding_sub2 = list()
""" Sentiment """
positive = set()
negative = set()
neutral = set()
""" Synonym """
synonym_dict = dict()
""" POS """
# concept_pos_combined_dict.pkl: dict[concept_seg]:concept_pos
conceptnet_pos_dict_path = CURRENT_PATH / r"data\conceptnet_pos_combined_dict.pkl"
conceptnet_pos_combined_dict = dict()
""" concept to concept_segs """
# concept2seg_dict.pkl: dict[concept]:concept_segs
concept2seg_dict_path = CURRENT_PATH / r'data\concept2seg_dict.pkl'
concept2seg_dict = dict()
""" NN """
NN_model = load_model(CURRENT_PATH / r"model\sentence_5_train_SENTLENGTH_10_replaced_concept_in_conceptnet_re.9_min_replaced_sent_2_hu256_dr.20_HAN.h5", custom_objects={"AttentionWithContext": AttentionWithContext, "LayerNormalization":LayerNormalization})
embedding_model = 0
WE_DIMENSIONS = 0
word2idx = dict()
idx2word = dict()
file_w = open(CURRENT_PATH / r"output\mcts_.txt", 'w', encoding="UTF-8")

invalid_sent_num = 0
total_paragraph_num = 0
invalid_sent_list = list()

GENERATED_PARAGRAPH_DIR_PATH = CURRENT_PATH / r"output"
TEMPLATES = ["Template_1", "Template_2", "Template_4", "Template_5", "Template_6", "Template_7", "Template_8"] # select one of the templates(template_3 is excluded)
# TEMPLATES = ["Template_9"]
MAX_PARAGRAPH_NUM = 10
MAX_SYNONYM_PARAGRAPH_NUM = 10
SENTIMENT_FORWARD_STEP = 0.9   # concept with the same sentiment will move forward in list of concepts
simulate_with_WE_step = 0
jieba.set_dictionary(CURRENT_PATH / r'data\PTT_dict.csv')

def main():
    global MCTS_TIME, conceptnet_pos_combined_dict, concept2seg_dict, simulate_with_WE_step

    # Load DB and model
    vocab_list_30 = readVocabList_freq(30)
    vocab_list_150 = readVocabList_freq(150)
    loadDB()
    conceptnet_pos_combined_dict = load_obj(conceptnet_pos_dict_path)
    concept2seg_dict = load_obj(concept2seg_dict_path)
    loadSynonyms(vocab_list_150)
    loadPretrainedEmbedding(vocab_list_30)

    backup_MCTS_TIME = MCTS_TIME
    change_start_concept_flag = False
    change_start_concept_count = 0
    paragraph_count = 0

    while paragraph_count < MAX_PARAGRAPH_NUM:
        synonyms = list()
        concepts = [["" for x in range(3)] for y in range(MAX_SENTENCE)]
        step = sentence = current_concept = ref_sentence = ref_concept = 0
        selected_node = list()
        root = Node()
        current_node = Node()
        next_node = Node()

        templateInitialize()

        # random.shuffle(TEMPLATES)  #需要random sort時解開
        schema = copy.deepcopy(TEMPLATES)
        getattr(sys.modules[__name__], schema[0])()

        # select initial concept from ConceptNet automatically
        start_concept = random.choice(CN_start_concept)
        CN_start_concept.remove(start_concept)
        
        # select initial concept manually
        # start_concept = "狗"
        # print("start concept:", start_concept)
        
        root.set_data(start_concept)
        if start_concept in synonym_dict:
            synonyms = findSynonyms(start_concept)

        print("Total Step:", totalStep())
        step = 0
        while step < totalStep():
            # reset template and mode
            getattr(sys.modules[__name__], schema[0])()
            simulate_with_WE_step = int(round(totalStep() / 2, 0))  #5.15

            # initialization
            if step == 0:
                selected_node.clear()
                concepts = [["" for x in range(3)] for y in range(MAX_SENTENCE)]
                current_node.resetNode()
                next_node.resetNode()

                selected_node.append(root)
                concepts[0][0] = root.data()
                current_node = copy.deepcopy(root)
                MCTS_TIME = backup_MCTS_TIME

            # update node's info to global mode and template
            sent = 0
            for node_ in selected_node:
                if node_.mode() != -1:
                    mode[sent] = node_.mode()
                    sent += 1
            for i, node_ in enumerate(selected_node):
                sent = currentSentence(i)
                current_concept_ = currentConcept(i)
                if current_concept_ == 1:
                    continue
                template[sent][current_concept_-2] = node_.relation()

                if template[sent][0] == "":
                    pdb.set_trace()
                    print()

                if node_.mode() == 2:
                    template[sent][1] = ""
                    continue

            sentence = currentSentence(step)
            current_concept = currentConcept(step)
            print("Root(第", sentence, "句的C", current_concept, "):", current_node.data(), sep="")

            # main裡的concepts和node新增順序是相同的
            # 如果是compound且是C3 或 single且是C2 時要跳過，因為下一步就是下一句的C1了，直接引用之前的concept，不再進行MCTS
            if (mode[sentence] != 2 and current_concept == 3) or (mode[sentence] == 2 and current_concept == 2):
                reset_concepts = copy.deepcopy(concepts)
                swapConcepts(concepts)
                ref_sentence = int(assoc_flag[sentence+1][0][0])  # sentence加1是因為需要的是下一句的assoc_flag資訊
                ref_concept = int(assoc_flag[sentence+1][0][1])
                print("第", (sentence+1), "句引用concepts[", ref_sentence, "][", ref_concept, "]:[", concepts[ref_sentence][ref_concept], "]當作C1", sep="")
                next_node.set_data(concepts[ref_sentence][ref_concept])
                concepts = reset_concepts
                concepts[sentence+1][0] = next_node.data()	     # currentConcept回傳值，1:C1, 2:C2, 3:C3
            else:
                next_node = MCTS(current_node, step, concepts, schema)
                concepts[sentence][current_concept] = next_node.data()

            if not next_node.data():
                sys.stderr.write("MCTS need more time to explore\n")
                sys.exit(-1)
            else:
                selected_node.append(Node(next_node))

            # expansion or simulation fails
            if next_node.data() == "change_schema":
                templateInitialize()
                # replace start concept with synonyms
                if synonyms:
                    print("-----Replace the start concept with synonyms-----")
                    random.shuffle(synonyms)
                    root.set_data(synonyms[0])
                    synonyms.remove(synonyms[0])
                    getattr(sys.modules[__name__], schema[0])()       # reset current template
                else:
                    change_start_concept_flag = changeSchema(schema)  # delete the template from the schema list and change to the others
                    if change_start_concept_flag:
                        change_start_concept_count += 1
                        if change_start_concept_count > 100:
                            print(start_concept, "沒有可以配對的template")
                            sys.exit(-1)
                        break
                    # change schema後找synonyms
                    if start_concept in synonym_dict:
                        synonyms = findSynonyms(start_concept)
                step = 0
                continue

            current_node.resetNode()
            # current_node = Node(next_node)         # without tree reuse
            current_node = copy.deepcopy(next_node)  # tree reuse
            next_node.resetNode()
            printConcepts(selected_node)
            step += 1

            file_w.write("step:"+str(step)+'\n')
            file_w.write("MCTS_TIME:"+str(MCTS_TIME)+'\n')
            if totalStep() - step < 3:
                MCTS_TIME = 10
            elif step % 4 == 0:
                MCTS_TIME /= 1.5

            print("-------------------NEXT ROUND-------------------")

        if change_start_concept_flag:
            change_start_concept_flag = False
            continue

        # print all selected nodes
        print("All selected nodes:")
        printConcepts(selected_node)
        printFinalResult(selected_node, schema)

        print("\ntotal predicted paragraph count:", total_paragraph_num)
        print("invalid_sent_num:", invalid_sent_num)
        print("invalid_sent:")
        for sent in invalid_sent_list:
            print(sent)

        paragraph_count += 1

    print("change start concept count:", change_start_concept_count)
    file_w.close()


#----------------------------------------------------------------------MCTS----------------------------------------------------------------------
# Monte Carlo Tree Search algorithm
def MCTS(node, current_step, past_concepts, schema):
    global mode, template
    expand_status = -1
    depth = delete_status = sentence = current_concept = score = 0
    current_node = Node()
    node_to_explore = Node()
    node_to_simulate = Node()
    selected_node = Node()

    # copy old data
    concepts = copy.deepcopy(past_concepts)
    mode_copy = copy.deepcopy(mode)
    template_copy = copy.deepcopy(template)

    start_time = time.time()
    while time.time() - start_time < MCTS_TIME:  # time.time() return current time(second)
        # reset Template function, the relation during simulation phase will not be saved
        getattr(sys.modules[__name__], schema[0])()

        # 避免已經固定的node因為template reset而把mode的資訊蓋掉
        concepts = copy.deepcopy(past_concepts)
        mode = copy.deepcopy(mode_copy)
        template = copy.deepcopy(template_copy)
        current_node = node
        depth = current_step  # reset depth

        print("Selection:(Root:第", currentSentence(current_step), "句的C", currentConcept(current_step), ")\n", current_node.data(), sep="")

        while not current_node.isLeaf():
            sentence = currentSentence(depth)
            current_concept = currentConcept(depth)

            if not current_node.firstSelection():  # 只有node第一次進入Selection時根據target_sentiment改變unvisited children nodes的順序，和target_sentiment相同的unvisited children nodes有較高的機率被select
                target_sentiment = targetSentiment(concepts, depth+1)
                node_to_explore = Selection(current_node, target_sentiment)
                current_node.setFirstSelection(True)
            else:
                node_to_explore = Selection(current_node, "")

            updateSurfacetext(depth, concepts, current_node, node_to_explore)

            # different C2 nodes have different mode
            if node_to_explore.mode() != -1:
                mode[sentence] = node_to_explore.mode()

            # 先判斷要select的node relation有沒有更新，有更新的話，也將template function更新，即便在同一層，node的relation也有可能不一樣
            if current_concept == 1 or (current_concept == 2 and mode[sentence] != 2):
                if template[sentence][current_concept-1] != node_to_explore.relation():
                    updateTemplate(sentence, node_to_explore.relation(), current_concept+1)
            # C2且end mode時，將第二個relation設為空
            if mode[sentence] == 2 and current_concept == 2:
                template[sentence][current_concept-1] = ""

            #test
            if mode[sentence] == 2 and concepts[sentence][2] != "":
                pdb.set_trace()
            #test

            # save selected nodes
            if (mode[sentence] != 2 and current_concept == 3) or (mode[sentence] == 2 and current_concept == 2):
                concepts[sentence+1][0] = node_to_explore.data()
                # print(node_to_explore.data()+" is used")
            else:
                concepts[sentence][current_concept] = node_to_explore.data()
                # print(node_to_explore.data()+" is selected")
            current_node = node_to_explore
            depth += 1  # a select action will increase the depth
        if depth == totalStep():
            current_node.set_terminal(True)

        reset_concepts = copy.deepcopy(concepts)
        swapConcepts(concepts)

        node_to_simulate = current_node
        if current_node.isVisited():
            expand_status = Expansion(current_node, depth, concepts)
            # 0:successful, -1:terminal, -99:no associate concept(that concept can't be used)
            if expand_status == 0 or expand_status == -99:
                concepts = copy.deepcopy(reset_concepts)
                continue
            if expand_status == -100:  # need to change the schema
                return Node("change_schema")

        score = Simulation(node_to_simulate, depth, concepts, schema)
        if score == -99:		# the path can't be used
            current_node = node_to_simulate.parent()
            # delete the node or nodes on upper layer
            delete_status = deleteUselessNode(node_to_simulate, depth)
            if delete_status == -100:
                return Node("change_schema")
            # simulation失敗，刪除該node後，沒有其他siblings，換expand backup_template嘗試
            if current_node and not current_node.children():
                if backup_template[sentence][0]:
                    updateTemplateAndMode(sentence)
                    expand_status = Expansion(current_node, depth-1, concepts)
                    if expand_status == 0 or expand_status == -99:
                        concepts = copy.deepcopy(reset_concepts)
                        continue
                    if expand_status == -100:  # need to change the schema
                        return Node("change_schema")
        elif score == -100:
            return Node("change_schema")
        else:  # simulate successfully
            backPropogation(node_to_simulate, score)
            # print("MCT:")
            # node.display(node, '')
        concepts = copy.deepcopy(reset_concepts)
        print("MCTS_TIME:", MCTS_TIME)

    print("Root:", node.data(), "(", node.visit_count(), ")", sep="")
    target_sentiment = targetSentiment(concepts, current_step+1)
    selected_node = node.selectChildren(target_sentiment)

    if not selected_node.data():
        sys.stderr.write("MCTS doesn't select any child node\n")
        pdb.set_trace()
        sys.exit(-1)

    print("MCTS select:", selected_node.data())
    updateSurfacetext(current_step, concepts, node, selected_node)

    return selected_node


"""
UCB1 = si + 2α*√lnN(v)/N(vi)  (si = Q(vi)/N(vi))
si:vi的平均分數
α:可調整參數，介於0~1
N(v):vi's parent被訪問次數
N(vi):vi節點被訪問次數
"""
# Select the node with highest UCT value
def Selection(node, target_sentiment):
    # print("----------------Selection---------------")
    uct_value = 0
    alpha = 0.25
    max_ = -100
    selected_node = Node()

    children_list = node.children()
    children_list, _ = sameSentiment(children_list, target_sentiment)  # select the same sentiment node first(if unvisited)

    for child in children_list:
        if not child.isVisited():  # select the node which hasn't been visited first
            print(child.data(), "is unvisited. Select it")
            return child
        si = child.score()/child.visit_count()
        uct_value = si + 2*alpha*sqrt(2*log(child.parent().visit_count()) / child.visit_count())
        # print("---"+child.data()+"("+str(child.visit_count())+", UCT:"+str(round(uct_value, 3))+", mean score:"+str(round(si, 3))+')')

        if uct_value > max_:
            max_ = uct_value
            selected_node = child
    print(selected_node.data(), "is selected")
    return selected_node


# Return value, 0:successful, -1:terminal, -99:no associate concpet(that concept can't be used), -100:change schema
# Add all associated nodes
def Expansion(node, depth, concepts):
    sentence = currentSentence(depth)
    current_concept = currentConcept(depth)
    c1_search_position_to_c2 = c1_search_position_to_c3 = ''
    swap_flag = False
    buf = list()

    if current_concept > 3:
        pdb.set_trace()
        print(depth)
        print(current_concept)

    print("----------------Expansion---------------")
    print("第", sentence, "句的 C", current_concept, ":", node.data(), " expand", sep="")
    if node.isTerminal():
        print("Error(Expansion).", node.data(), "is a terminal node. Can't expand it!!")
        return -1

    # 如果是compound且是C3 或 single且是C2 時要跳過，因為下一步就是下一句的C1了，C1直接引用之前的concept
    if (mode[sentence] != 2 and current_concept == 3) or (mode[sentence] == 2 and current_concept == 2):
        ref_sentence = int(assoc_flag[sentence+1][0][0])
        ref_concept = int(assoc_flag[sentence+1][0][1])
        c1 = concepts[ref_sentence][ref_concept]
        print(node.data(), "expand C1:", c1)
        node.addChild(Node(c1))
        return 0

    rel = template[sentence][current_concept-1]
    # single relation
    if mode[sentence] == 2:
        c1 = node.data()
    # compound relation
    else:
        match_str = MATCH_TABLE[Convert(template[sentence][0])][Convert(template[sentence][1])]
        if Convert(template[sentence][0]) > Convert(template[sentence][1]):
            swap_flag = True

        # 1:start, 2:end, 3:both(C1 may be in the Start or End position)
        c1_search_position_to_c2 = match_str[0]
        c1_search_position_to_c3 = match_str[1]
        if c1_search_position_to_c2 == '3':
            c1_search_position_to_c2 = compound_pos[sentence][0]
        if c1_search_position_to_c3 == '3':
            c1_search_position_to_c3 = compound_pos[sentence][1]
        if swap_flag:                # reverse relations
            c1_search_position_to_c2, c1_search_position_to_c3 = c1_search_position_to_c3, c1_search_position_to_c2

        if current_concept == 1:     # c1 expand c2
            if mode[sentence] == 1:
                c1_search_position_to_c2 = invertStartEnd(c1_search_position_to_c2)
            c1 = node.data()
        else:                        # c2 expand c3
            if mode[sentence] == 1:  # end mode時，tree的順序為C2-C1-C3，C1在第二層
                c1 = node.data()
            else:			       # 假如要expand的node是C2，C2的parent是C1
                c1 = concepts[sentence][0]

    # C1's serach position is in End
    c1_search_position = "End"
    # C1's serach position is in Start
    if c1_search_position_to_c2 == '1' or c1_search_position_to_c3 == '1' or single_pos[sentence] == "Start":
        c1_search_position = invertStartEnd(c1_search_position)

    buf = selectConceptFromDB(c1, c1_search_position, rel, buf)

    # delete duplicate
    for concept in buf[:]:
        duplicate_depth, _ = duplicateCheck(concept, concepts, sentence, current_concept)
        if duplicate_depth != -1:
            buf.remove(concept)

    # 透過association chcek刪除不適用的node  6.6
    if assoc_flag[sentence][current_concept]:
        assoc_sentence = int(assoc_flag[sentence][current_concept][0])
        assoc_concept = int(assoc_flag[sentence][current_concept][1])
        target_concept = concepts[assoc_sentence][assoc_concept]
        related_concepts_dict = assocConcepts(target_concept, buf)
        zero_score_num = 0
        print("cosine score:")
        for concept, score in related_concepts_dict.items():
            if score == 0:
                zero_score_num += 1
            print(concept, score)
        print("total_num:", len(related_concepts_dict))
        print("zero_num:", zero_score_num)
        ratio = round(zero_score_num / len(related_concepts_dict), 3)
        print("ratio:", ratio)
        print("target_concept, concepts[%d][%d]:%s" % (assoc_sentence, assoc_concept, target_concept))
        print(concepts)
        print(template)
        if sentence == 2 and current_concept == 1:
            # pdb.set_trace()
            pass
        if ratio > 0.7:
            buf.clear()

    for concept in buf:
        node.addChild(Node(concept))

    # 找不到possible moves時，代表該concept透過某個relation找不到associate concept，此時要把該concept及其相關的 parent concept刪除掉，確保之後不會再走到這一條路
    if not node.children():
        print("Error(Expansion).第", sentence, "句的C", current_concept, " can't expand. Try to use backup_template", sep="")
        while backup_template[sentence][0]:
            updateTemplateAndMode(sentence)
            if Expansion(node, depth, concepts) == 0:
                return 0
        delete_status = deleteUselessNode(node, depth)
        if delete_status == -100:
            return -100
        return -99

    if c1_search_position_to_c2 == '1' or c1_search_position_to_c3 == '1' or single_pos[sentence] == "Start":
        for child in node.children():
            child.set_search_position("End")
    elif c1_search_position_to_c2 == '2' or c1_search_position_to_c3 == '2' or single_pos[sentence] == "End":
        for child in node.children():
            child.set_search_position("Start")

    # add relation info to C2 or C3
    for child in node.children():
        child.set_relation(rel)

    # set mode to C2
    if current_concept == 1:
        for child in node.children():
            child.set_mode(mode[sentence])
    elif current_concept == 2:
        node.set_mode(mode[sentence])

    return 0


# Simulate randomly and return the score
def Simulation(node, depth, past_concepts, schema):
    global c2_reselect_flag, c3_reselect_flag, mode, template
    c2_reselect_flag = c3_reselect_flag = False
    match = sentence = total_sentence = root_sentence = 0
    ref_sentence = ref_concept = -1
    start_sentence = simulated_concept = 0
    duplicate_depth = tmp_mode = -1
    match_status = swap_flag = False
    end_to_single = np.zeros((MAX_SENTENCE, ), dtype=bool)
    c1 = c2 = c3 = rel_1 = rel_2 = ""
    c1_search_position = c2_search_position = surfacetext_1 = surfacetext_2 = match_str = ""
    c1_search_position_to_c2 = c1_search_position_to_c3 = ""
    concepts = [["" for x in range(3)] for y in range(MAX_SENTENCE)]
    reselect_times = 0

    print("----------------Simulation--------------")
    total_sentence = templateSentenceNum()
    start_sentence = currentSentence(depth)
    simulated_concept = currentConcept(depth)
    print("total_sentence:", total_sentence)
    print("start_sentence:", start_sentence)
    print("depth:", depth)
    print("simulated_concept: C", simulated_concept, "[", node.data(), "]", sep="")

    buf_c2 = list()
    for i in range(total_sentence):
        buf_c2.append([])
    buf_c3 = list()
    for i in range(total_sentence):
        buf_c3.append([])

    # copy old data
    concepts = copy.deepcopy(past_concepts)
    mode_copy = copy.deepcopy(mode)
    template_copy = copy.deepcopy(template)

    root_sentence = node.getRootSentence(depth)
    # sentence要根據simulation的起點做調整
    sentence = start_sentence
    while sentence < total_sentence:
        c1 = c2 = c3 = ""
        swap_flag = False
        concepts[sentence+1:] = [["" for x in range(3)] for y in range(MAX_SENTENCE-sentence-1)]  # 清空該句後面的concepts，避免一些判斷上的干擾

        print("\n第", sentence, "句", sep="", end="")
        if mode[sentence] == 1:
            print("(End mode)")
        elif mode[sentence] == 2:
            print("(Single mode)")
        else:
            print()

        # if c1 has association with previous concept
        if assoc_flag[sentence][0]:
            ref_sentence = int(assoc_flag[sentence][0][0])
            ref_concept = int(assoc_flag[sentence][0][1])
            c1 = concepts[ref_sentence][ref_concept]
        else:
            if (mode[sentence] == 1 and concepts[sentence][1] != "") or end_to_single[sentence]:
                c1 = concepts[sentence][1]
            else:
                c1 = concepts[sentence][0]
            end_to_single[sentence] = False

        # Compound relation mode(mode為0或1)
        if mode[sentence] != 2:
            rel_1 = template[sentence][0]
            rel_2 = template[sentence][1]

            # Avoid duplicate compound relation(ex:CapableOf-AtLocation can find in AtLocation-CapableOf)
            if Convert(rel_1) > Convert(rel_2):
                swap_flag = True
            print("Relation: ", rel_1, "-", rel_2, sep="")

            # To determine the search concept is in start or end according to the compound relation
            match_str = MATCH_TABLE[Convert(rel_1)][Convert(rel_2)]

            """
            0:(end, end)
            1:(end, start)
            2:(start, end)
            3:(start, start)
            """
            match = Match(match_str, swap_flag, sentence)
            c1_search_position_to_c2 = "Start"
            c1_search_position_to_c3 = "Start"
            if match == 0 or match == 1:
                c1_search_position_to_c2 = "End"
            if match == 0 or match == 2:
                c1_search_position_to_c3 = "End"

            # if start concept is end node, swap the search position
            if mode[sentence] == 1:
                c1_search_position_to_c2 = invertStartEnd(c1_search_position_to_c2)

            # if it is the first loop and the simulated node is c2, assign the node's data to c2 directly
            if (sentence == start_sentence) and (simulated_concept == 2):
                c2 = node.data()
                print("Start concept:", c1)
                print("Simulated node is C2[", c2, "]", sep="")

                if template[sentence][0] == node.relation():
                    surfacetext_1 = selectSurfacetext(c1, c2, node.search_position(), rel_1)
                    print("SurfaceText_1:", surfacetext_1)
                # relation有改變
                elif template[sentence][0] != node.relation():
                    rel_1 = template[sentence][0]
                    print("template[", sentence, "][0]:\"", node.relation(), "\"→", rel_1, sep="")
                    print("檢驗 C1-", rel_1, "-", c2, " 是否存在", sep="")
                    selectConceptFromDB(c1, c1_search_position_to_c2, rel_1, buf_c2, sentence)
                    if c2 in buf_c2[sentence]:
                        print("C1-", rel_1, "-", c2, " 存在!", sep="")
                        node.set_relation(rel_1)  # 更新這個node的relation
                        buf_c2[sentence].clear()
                    else:
                        buf_c2[sentence].clear()
                        if backup_template[sentence][0]:
                            print("Error(Simulation→relation fail). relation 1:\"", rel_1, "\" in sentence ", sentence, " can't be used", sep="")
                            updateTemplateAndMode(sentence)
                            continue
                        print("Error(Simulation). The simulated_node[", node.data(), "] can't be used", sep="")
                        return -99

                # if start concept is end node, swap the end node and the search node. To make sure c1 is the search node
                if mode[sentence] == 1:
                    c1, c2 = c2, c1
                saveConcepts(concepts, c1, c2, c3, sentence)

            elif (sentence == start_sentence) and (simulated_concept == 3):
                c1 = concepts[sentence][0]
                c2 = concepts[sentence][1]
                c3 = concepts[sentence][2]

                if Convert(rel_1) > Convert(rel_2) and concepts[sentence][2]:
                    c2, c3 = c3, c2

                if mode[sentence] == 1:
                    print("Start concept:", c2)
                else:
                    print("Start concept: ", c1)
                print("Simulated node is C3[", c3, "]", sep="")

                if mode[sentence] == 1:
                    surfacetext_1 = selectSurfacetext(c2, c1, node.parent().search_position(), rel_1)
                else:
                    surfacetext_1 = selectSurfacetext(c1, c2, node.parent().search_position(), rel_1)
                print("SurfaceText_1:", surfacetext_1)

                surfacetext_2 = selectSurfacetext(c1, c3, node.search_position(), rel_2)
                print("SurfaceText_2:", surfacetext_2)

                if Convert(rel_1) > Convert(rel_2) and concepts[sentence][2]:
                    c2, c3 = c3, c2
                    rel_1, rel_2 = rel_2, rel_1
                    match = Match(match_str, False, sentence)
                Generate(c1, c2, c3, rel_1, rel_2, surfacetext_1, surfacetext_2, sentence, match)
                printConcepts(concepts, sentence)
                sentence += 1
                continue

            else:
                print("Start concept:", c1)
                if not c3_reselect_flag:
                    if not c2_reselect_flag:
                        selectConceptFromDB(c1, c1_search_position_to_c2, rel_1, buf_c2, sentence)  # Search c2 according to c1 and relation_1
                    if not buf_c2[sentence]:
                        print("Error(Simulation). ", c1, "(", rel_1, ", ", c1_search_position_to_c2, ") can't find match c2!", sep="")
                        if backup_template[sentence][0]:
                            print("Error(Simulation→relation fail). relation 1:\"", rel_1, "\" in sentence ", sentence, " can't be used", sep="")
                            updateTemplateAndMode(sentence)
                        else:
                            # 原本的ref_concept有可能是交換(end mode或reverse relation)後的狀態，如果要回到該句做一些處理，需要把ref_concept轉回該句使用時的狀態
                            ref_concept_before_swapping = resetConceptsPosition(ref_sentence, ref_concept)
                            # 如果此句是root_sentence 或 使用的是root_sentence的start concept，就無法再更換成其他concept了，需要替換另一個schema
                            if sentence == root_sentence or ref_concept_before_swapping == 0:
                                sys.stderr.write("Error(Simulation). Change the schema\n")
                                return -100  # 回傳一個simulation評分分數不會出現的數字
                            # 引用的句子小於起始句(不在simulation範圍)，或引用起始句但concept小於simulated node，代表需要更換c1
                            if Depth(ref_sentence, ref_concept_before_swapping) <= depth:
                                print("Error(Simulation). The node [", ref_sentence, "][", ref_concept, "][", concepts[ref_sentence][ref_concept], "] used by start concept [", c1, "] is a fixed data", sep="")
                                return -99
                            # 回到c1引用的那一句，並刪除buf裡找不到的那個concept，下次選擇時就不會再選到(例如第4句的c1引用了第3句的cy，便回到第3句重新選擇)
                            deleteUselessConcept(c1, ref_sentence, ref_concept_before_swapping, buf_c2, buf_c3)
                            sentence = ref_sentence
                            #只用copy覆蓋到x+1句，x及之前的句子保持原樣
                            for i in range(ref_sentence+1, MAX_SENTENCE):
                                mode[i] = mode_copy[i]
                                template[i][0] = template_copy[i][0]
                                template[i][1] = template_copy[i][1]
                        continue
                    # print("Concept_2 related to ", c1, "(", rel_1, ", ", c1_search_position_to_c2, ")(", len(buf_c2[sentence]), "): ", buf_c2[sentence])

                    # Select a C2
                    # 在step < N 時才使用 simulate with CWE(Count-based Word Embedding)(在中早期branch比較多時才需要用此策略來找到相對精準的simulation結果)
                    if depth < simulate_with_WE_step and assoc_flag[sentence][1]:
                        assoc_sentence = int(assoc_flag[sentence][1][0])
                        assoc_concept = int(assoc_flag[sentence][1][1])
                        target_concept = concepts[assoc_sentence][assoc_concept]
                        c2 = selectAssocConcept(target_concept, buf_c2[sentence])
                    if not c2:
                        c2 = random.choice(buf_c2[sentence])

                    concepts[sentence][0] = c1
                    concepts[sentence][1] = ""

                    # avoid duplicate concepts
                    duplicate_depth, duplicate_concept = duplicateCheck(c2, concepts, sentence, 1)
                    print("duplicate_depth:", duplicate_depth)
                    if duplicate_depth != -1:
                        buf_c2[sentence].remove(c2)
                        if not buf_c2[sentence]:
                            print("C2:", c2, " 重複，且沒有其他siblings", sep="")
                            removing_concept, back_to_sentence, back_to_concept = backToSentence(c1, duplicate_concept, sentence, duplicate_depth, buf_c2, buf_c3)
                            print("back_to_sentence:", back_to_sentence)
                            print("back_to_concept:", back_to_concept)
                            if sentence == root_sentence or back_to_concept == 0:
                                sys.stderr.write("Error(Simulation→duplicateCheck). Change the schema\n")
                                return -100
                            if Depth(back_to_sentence, back_to_concept) <= depth:
                                print("Error(Simulation→duplicateCheck). The node ", removing_concept, "第", back_to_sentence, "句的C", back_to_concept+1, "] is a fixed data", sep="")
                                return -99
                            deleteUselessConcept(removing_concept, back_to_sentence, back_to_concept, buf_c2, buf_c3)
                            sentence = back_to_sentence
                            #只用copy覆蓋到x+1句，x及之前的句子保持原樣
                            for i in range(back_to_sentence+1, MAX_SENTENCE):
                                mode[i] = mode_copy[i]
                                template[i][0] = template_copy[i][0]
                                template[i][1] = template_copy[i][1]
                        else:
                            print("C2:", c2, " 重複，但還有其他siblings(", len(buf_c2[sentence]), "個)", sep="")
                            c2_reselect_flag = True
                        continue

                    if (not concepts[sentence][2]) or (mode[sentence] == 1):  # 當c3還未選擇 或 end mode時C2重選，reset c2_reselect_flag，可以繼續選擇c3
                        c2_reselect_flag = False

                    # print C1-C2's SurfaceText
                    surfacetext_1 = selectSurfacetext(c1, c2, invertStartEnd(c1_search_position_to_c2), rel_1)
                    print("SurfaceText_1:", surfacetext_1)

                    if mode[sentence] == 1:
                        c1, c2 = c2, c1
                    saveConcepts(concepts, c1, c2, c3, sentence)
                else:
                    # reverse relation時交換
                    if Convert(rel_1) > Convert(rel_2) and concepts[sentence][2]:
                        concepts[sentence][1], concepts[sentence][2] = concepts[sentence][2], concepts[sentence][1]
                    c1 = concepts[sentence][0]
                    c2 = concepts[sentence][1]  # C3重找，C2沒變，所以直接提取儲存在concepts裡面的值

            if not c2_reselect_flag:
                # Search c3 according to c1 and relation_2
                if not c3_reselect_flag:
                    selectConceptFromDB(c1, c1_search_position_to_c3, rel_2, buf_c3, sentence)

                if not buf_c3[sentence]:
                    # 非第一次(sentence!=0)C3為空時，根據concepts[ref_sentence][right]來刪除buf_c3[sentence][]裡的那個concept，如果buf_c3刪除完為空→換template
                    print("Error(Simulation). ", c1, "(", rel_2, ", ", c1_search_position_to_c3, ") can't find match c3!", sep="")

                    # end mode且buf_c2還有超過1個以上可以汰換時(C1為原本的C2，所以只要重找C2就可以改變C1)
                    if (mode[sentence] == 1) and (len(buf_c2[sentence]) > 1):
                        deleteUselessConcept(c1, sentence, 1, buf_c2, buf_c3)
                    elif backup_template[sentence][0]:
                        # 其他句的C1都是引用別句的，只有第0句的C1是直接給的
                        if sentence == 0 and mode[sentence] == 1 and backup_mode[sentence] == 2:
                            end_to_single[sentence] = True

                        tmp_mode = -1
                        if sentence == start_sentence and simulated_concept == 2:
                            tmp_mode = mode[sentence]
                        print("Error(Simulation→relation fail). relation 2:\"", rel_2, "\" in sentence ", sentence, " can't be used", sep="")
                        updateTemplateAndMode(sentence)

                        # 如果update後mode有變化，也改變node的mode
                        if tmp_mode != -1 and tmp_mode != mode[sentence]:
                            # pdb.set_trace()
                            node.set_mode(mode[sentence])
                    # end mode且要刪除的是simulated node時，離開並刪除此node
                    elif mode[sentence] == 1 and (sentence == start_sentence) and (simulated_concept == 2):  # C2是simulated node，是已固定的
                        print("Error(Simulation). ", node.data(), "(End mode) can't find associated concepts", sep="")
                        return -99
                    else:
                        ref_concept_before_swapping = resetConceptsPosition(ref_sentence, ref_concept)
                        if sentence == root_sentence or ref_concept_before_swapping == 0:
                            sys.stderr.write("Error(Simulation). Change the schema\n")
                            return -100
                        if Depth(ref_sentence, ref_concept_before_swapping) <= depth:
                            print("Error(Simulation). The node [", ref_sentence, "][", ref_concept, "][", concepts[ref_sentence][ref_concept], "] used by start concept [", c1, "] is a fixed data", sep="")
                            return -99
                        if mode[sentence] == 1:
                            deleteUselessConcept(c2, ref_sentence, ref_concept_before_swapping, buf_c2, buf_c3)	    # end mode時，C2是start concept
                        else:
                            deleteUselessConcept(c1, ref_sentence, ref_concept_before_swapping, buf_c2, buf_c3)
                        sentence = ref_sentence
                        #只用copy覆蓋到x+1句，x及之前的句子保持原樣
                        for i in range(ref_sentence+1, MAX_SENTENCE):
                            mode[i] = mode_copy[i]
                            template[i][0] = template_copy[i][0]
                            template[i][1] = template_copy[i][1]

                    continue
                # print("Concept_3 related to ", c1, "(", rel_2, ", ", c1_search_position_to_c3, ")(", len(buf_c3[sentence]), "): ", buf_c3[sentence], sep="")

                # Select a c3
                if depth < simulate_with_WE_step and assoc_flag[sentence][2]:
                    assoc_sentence = int(assoc_flag[sentence][2][0])
                    assoc_concept = int(assoc_flag[sentence][2][1])
                    target_concept = concepts[assoc_sentence][assoc_concept]
                    c3 = selectAssocConcept(target_concept, buf_c3[sentence])
                if not c3:
                    c3 = random.choice(buf_c3[sentence])

                concepts[sentence][2] = ""

                # avoid duplicate concepts
                duplicate_depth, duplicate_concept = duplicateCheck(c3, concepts, sentence, 2)
                print("duplicate_depth:", duplicate_depth)
                if duplicate_depth != -1:
                    buf_c3[sentence].remove(c3)
                    # 每一個都和前面的concepts重複了
                    if not buf_c3[sentence]:
                        if mode[sentence] == 1 and len(buf_c2[sentence]) > 1:
                            print("end mode時，C3刪到只剩重複的，重選同一句的C2")
                            deleteUselessConcept(c1, sentence, 1, buf_c2, buf_c3)
                        elif mode[sentence] == 1 and (sentence == start_sentence) and (simulated_concept == 2):  # C2是已固定的simulated node
                            print("Error(Simulation). ", node.data(), "(End mode) can't find associated concepts", sep="")
                            return -99
                        else:
                            print("C3:", c3, " 重複，且沒有其他siblings", sep="")
                            if mode[sentence] == 1:
                                removing_concept, back_to_sentence, back_to_concept = backToSentence(c2, duplicate_concept, sentence, duplicate_depth, buf_c2, buf_c3)
                            else:
                                removing_concept, back_to_sentence, back_to_concept = backToSentence(c1, duplicate_concept, sentence, duplicate_depth, buf_c2, buf_c3)
                            print("back_to_sentence:", back_to_sentence)
                            print("back_to_concept:", back_to_concept)
                            if sentence == root_sentence or back_to_concept == 0:
                                sys.stderr.write("Error(Simulation→duplicateCheck). Change the schema\n")
                                return -100
                            if Depth(back_to_sentence, back_to_concept) <= depth:
                                print("Error(Simulation→duplicateCheck). The node ", removing_concept, "第", back_to_sentence, "句的C", back_to_concept+1, "] is a fixed data", sep="")
                                return -99
                            deleteUselessConcept(removing_concept, back_to_sentence, back_to_concept, buf_c2, buf_c3)
                            sentence = back_to_sentence
                            for i in range(back_to_sentence+1, MAX_SENTENCE):
                                mode[i] = mode_copy[i]
                                template[i][0] = template_copy[i][0]
                                template[i][1] = template_copy[i][1]
                    else:
                        print("C3:", c3, " 重複，但還有其他siblings(", len(buf_c3[sentence]), "個)", sep="")
                        c3_reselect_flag = True
                    continue

                # Print C1-C3's SurfaceText
                surfacetext_2 = selectSurfacetext(c1, c3, invertStartEnd(c1_search_position_to_c3), rel_2)
                print("SurfaceText_2:", surfacetext_2)
            else:
                c3 = concepts[sentence][2]  # 在c2_reselect_flag為true的情況下，因為C3沒變，所以直接提取儲存在concepts裡面的值

            if swap_flag:
                c2, c3 = c3, c2
                rel_1, rel_2 = rel_2, rel_1
                surfacetext_1, surfacetext_2 = surfacetext_2, surfacetext_1
                match = Match(match_str, False, sentence)  # get correct match in normal relation order

            # test
            if c3 == "":
                pdb.set_trace()
            # test

            saveConcepts(concepts, c1, c2, c3, sentence)

            match_status = Generate(c1, c2, c3, rel_1, rel_2, surfacetext_1, surfacetext_2, sentence, match)
            if match_status:
                print(SurfaceText[sentence], '\n')
            else:
                # avoid infinite loop
                reselect_times += 1
                if reselect_times > 20:
                    sys.stderr.write("Error(Simulation). reselect_times > 20. Change the schema\n")
                    return -100
                sentence -= 1

        # Single relation mode
        else:
            rel_1 = template[sentence][0]
            print("Relation:", rel_1)

            # To determine the search concept is in start or end
            c1_search_position = "Start"
            c2_search_position = "End"
            if single_pos[sentence] == "End":
                c1_search_position, c2_search_position = c2_search_position, c1_search_position

            if sentence == start_sentence and simulated_concept == 2 and (template[sentence][0] == node.relation()):
                c2 = node.data()
                print("Start concept:", c1)
                print("Simulated node is C2[", c2, "]", sep="")
                surfacetext_1 = selectSurfacetext(c1, c2, c2_search_position, rel_1)
                SurfaceText[sentence] = surfacetext_1
                print("Single relation:", surfacetext_1)
                saveConcepts(concepts, c1, c2, c3, sentence)
                sentence += 1
                continue
            if sentence == start_sentence and simulated_concept == 2 and (template[sentence][0] != node.relation()):
                c2 = node.data()
                rel_1 = template[sentence][0]
                print("Start concept:", c1)
                print("Simulated node is C2[", c2, "]", sep="")
                print("template[", sentence, "][0]:\"", node.relation(), "\"→", rel_1, sep="")
                print("檢驗 C1-", rel_1, "-", c2, " 是否存在", sep="")
                selectConceptFromDB(c1, c1_search_position, rel_1, buf_c2, sentence)
                if c2 in buf_c2[sentence]:
                    print("C1-", rel_1, "", c2, " 存在!", sep="")
                    node.set_relation(rel_1)  # 更新這個node的relation
                    buf_c2[sentence].clear()
                    sentence += 1
                else:
                    buf_c2[sentence].clear()
                    if backup_template[sentence][0]:
                        print("Error(Simulation→relation fail). single relation:\"", rel_2, "\" in sentence ", sentence, " can't be used", sep="")
                        updateTemplateAndMode(sentence)
                    else:
                        print("Error(Simulation). The simulated_node[", node.data(), "] can't be used", sep="")
                        return -99
                continue

            print("Start concept:", c1)

            if not c2_reselect_flag:
                selectConceptFromDB(c1, c1_search_position, rel_1, buf_c2, sentence)

            # 參考compound relationc1-c2的情況做修改
            if not buf_c2[sentence]:
                print("Error(Simulation). ", c1, "(", rel_1, ") can't find match c2(single sentence)!", sep="")
                if backup_template[sentence][0]:
                    print("第", sentence, "句的single relation \"", rel_1, "\" 無法使用", sep="")
                    updateTemplateAndMode(sentence)
                    # print("New relation: ", template[sentence][0])
                else:
                    ref_concept_before_swapping = resetConceptsPosition(ref_sentence, ref_concept)
                    if sentence == root_sentence or ref_concept_before_swapping == 0:
                        sys.stderr.write("Error(Simulation). Change the schema\n")
                        return -100
                    if Depth(ref_sentence, ref_concept_before_swapping) <= depth:
                        print("Error(Simulation). The node [", ref_sentence, "][", ref_concept, "][", concepts[ref_sentence][ref_concept], "] used by start concept [", c1, "] is a fixed data", sep="")
                        return -99
                    deleteUselessConcept(c1, ref_sentence, ref_concept_before_swapping, buf_c2, buf_c3)
                    sentence = ref_sentence
                    #只用copy覆蓋到x+1句，x及之前的句子保持原樣
                    for i in range(ref_sentence+1, MAX_SENTENCE):
                        mode[i] = mode_copy[i]
                        template[i][0] = template_copy[i][0]
                        template[i][1] = template_copy[i][1]
                continue
            # print("Concept_2 related to ", c1, "(", rel_1, ", ", c1_search_position, ")(", len(buf_c2[sentence]), "): ", buf_c2[sentence])

            # Select a c2
            if depth < simulate_with_WE_step and assoc_flag[sentence][1]:
                assoc_sentence = int(assoc_flag[sentence][1][0])
                assoc_concept = int(assoc_flag[sentence][1][1])
                target_concept = concepts[assoc_sentence][assoc_concept]
                c2 = selectAssocConcept(target_concept, buf_c2[sentence])
            if not c2:
                c2 = random.choice(buf_c2[sentence])

            concepts[sentence][0] = c1
            concepts[sentence][1] = ""

            # avoid duplicate concept
            duplicate_depth, duplicate_concept = duplicateCheck(c2, concepts, sentence, 1)
            print("duplicate_depth:", duplicate_depth)
            if duplicate_depth != -1:
                buf_c2[sentence].remove(c2)
                if not buf_c2[sentence]:
                    print("C2:", c2, " 重複，且沒有其他siblings", sep="")
                    removing_concept, back_to_sentence, back_to_concept = backToSentence(c1, duplicate_concept, sentence, duplicate_depth, buf_c2, buf_c3)
                    print("back_to_sentence:", back_to_sentence)
                    print("back_to_concept:", back_to_concept)
                    if sentence == root_sentence or back_to_concept == 0:
                        sys.stderr.write("Error(Simulation→duplicateCheck). Change the schema\n")
                        return -100
                    if Depth(back_to_sentence, back_to_concept) <= depth:
                        print("Error(Simulation→duplicateCheck). The node ", removing_concept, "第", back_to_sentence, "句的C", back_to_concept+1, "] is a fixed data", sep="")
                        return -99
                    deleteUselessConcept(removing_concept, back_to_sentence, back_to_concept, buf_c2, buf_c3)
                    sentence = back_to_sentence
                    for i in range(back_to_sentence+1, MAX_SENTENCE):
                        mode[i] = mode_copy[i]
                        template[i][0] = template_copy[i][0]
                        template[i][1] = template_copy[i][1]
                else:
                    print("C2:", c2, " 重複，但還有其他siblings(", len(buf_c2[sentence]), "個)", sep="")
                    c2_reselect_flag = True
                continue

            # Print single relation SurfaceText
            surfacetext_1 = selectSurfacetext(c1, c2, c2_search_position, rel_1)
            SurfaceText[sentence] = surfacetext_1
            print("Single relation:", surfacetext_1)

            saveConcepts(concepts, c1, c2, c3, sentence)

        printConcepts(concepts, sentence)
        c2_reselect_flag = False
        c3_reselect_flag = False
        sentence += 1

    pred_paragraph = SurfaceText[:templateSentenceNum()]

    # adjust predicted paragraph according to schema(送到predictScore和最後呈現的pred_paragraph是不一樣的)
    if schema[0] == "Template_2":
        supplement = Supplement_MotivatedByGoal(concepts[4][1])
        pred_paragraph[3] = pred_paragraph[3] + supplement + concepts[4][1]
        pred_paragraph.pop(4)
    elif schema[0] == "Template_3":
        pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
        # pred_paragraph[2] = pred_paragraph[2] + '，會讓人想要 ' + concepts[3][1]
        pred_paragraph.pop(3)
    elif schema[0] == "Template_4":
        if mode[3] == 2:
            supplement = Supplement_MotivatedByGoal(concepts[3][1])
            pred_paragraph[2] = pred_paragraph[2] + supplement + concepts[3][1]
        elif mode[3] == 1:
            pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][2], '').split())
        pred_paragraph.pop(3)
        pred_paragraph.pop(3)
    elif schema[0] == "Template_5":
        if mode[2] == 2:
            pred_paragraph[2] = "為了 " + concepts[2][1] + " 而 " + concepts[2][0] + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
        else:
            pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
        pred_paragraph.pop(3)
        supplement = Supplement_MotivatedByGoal(concepts[5][1])
        pred_paragraph[3] = pred_paragraph[3] + supplement + concepts[5][1]
        pred_paragraph.pop(-1)
    elif schema[0] == "Template_7":
        supplement = Supplement_MotivatedByGoal(concepts[1][1])
        pred_paragraph[0] = pred_paragraph[0] + supplement + concepts[1][1]
        pred_paragraph.pop(1)
        pred_paragraph.pop(1)
        pred_paragraph.pop(3)
    elif schema[0] == "Template_8":
        if mode[3] == 2:
            supplement = Supplement_MotivatedByGoal(concepts[3][1])
            pred_paragraph[2] = pred_paragraph[2] + supplement + concepts[3][1]
        elif mode[3] == 1:
            pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][2], '').split())
        pred_paragraph.pop(3)
        pred_paragraph.pop(3)
    elif schema[0] == "Template_9":
        pred_paragraph.pop(2)
        pred_paragraph.pop(3)

    removeSurfacetextFormat(pred_paragraph)
    # remove brackets
    for i in range(len(pred_paragraph)):
        pred_paragraph[i] = pred_paragraph[i].replace('[', '')
        pred_paragraph[i] = pred_paragraph[i].replace(']', '')
        pred_paragraph[i] = ' '.join(pred_paragraph[i].split())

    # predict paragraph score
    score = predictScore(pred_paragraph)

    file_w.write('"' + '\n')
    for sent in pred_paragraph:
        file_w.write(sent + '\n')
    file_w.write("Score:" + str(score) + '\n')
    file_w.write("fixed node num:" + str(depth) + '\n')
    file_w.write('--"' + '\n')
    print()
    print(node.data(), "'s simulation score:", score)

    return score


# Update the score and visit count
def backPropogation(node, score):
    print("BackPropagation:")
    while node.parent():
        node.add_score(score)
        node.incrementVisitCount()
        print(" ", node.data(), "(", node.visit_count(), ")", sep="")
        node = node.parent()
    node.add_score(score)
    node.incrementVisitCount()
    print(" ", node.data(), "(", node.visit_count(), ")", sep="")


# return duplicate depth, -1 if duplicate not found
def duplicateCheck(checked_concept, concepts, sentence, current_concept):
    concepts = resetConceptsPosition(concepts, sentence, current_concept)
    depth = -1
    for i in range(sentence+1):
        for j, concept in enumerate(concepts[i]):
            if concept:
                depth += 1
                if concept2seg_dict[concept] == concept2seg_dict[checked_concept]:
                    print("Error(duplicateCheck). 第", sentence, "句的C", (current_concept+1), "[", checked_concept, "] is the same as concepts[", i, "][", j, "]:[", concept, "](after segmentation)", sep="")
                    return depth, concept
    return -1, ''


# 之前做法是針對c1去刪除，但發現有可能會導致下述程式無法順利結束
# 例如:模擬x點時沒有選到重複的，順利結束。x變成fixed data，在後續的模擬卻選到重複的，且需要刪除x，但因為x是fixed data無法刪除，造成程式停止
# 當buf[sentence]裡可選的concept只剩一個時，刪除與之重複的data，確保程式順利執行
def backToSentence(c1, duplicate_concept_str, sentence, duplicate_depth, buf_c2, buf_c3):
    print("---------------backToSentence--------------------")
    #第0句的C2、C3所引用的concept都沒有其他選擇了，也沒有辦法往前跳 或 和第0句的start concept重複 -> change schema
    if sentence == 0 or duplicate_depth == 0:
        return "", 0, 0

    ref_sentence = int(assoc_flag[sentence][0][0])
    ref_concept = int(assoc_flag[sentence][0][1])
    ref_concept = resetConceptsPosition(ref_sentence, ref_concept)
    # print("ref_sentence:", ref_sentence)
    # print("ref_concept:", ref_concept)

    #所引用的concept是一開始的start concept
    if ref_concept == 0:
        return "", 0, 0

    if ref_concept == 1:
        buf_ref = buf_c2
    elif ref_concept == 2:
        buf_ref = buf_c3
    ref_buf_count = len(buf_ref[ref_sentence])

    duplicate_sentence = currentSentence(duplicate_depth)
    duplicate_concept = currentConcept(duplicate_depth)-1
    # print("duplicate_sentence:", duplicate_sentence)
    # print("duplicate_concept:", duplicate_concept)
    if duplicate_concept == 1:
        buf_dup = buf_c2
    elif duplicate_concept == 2:
        buf_dup = buf_c3
    duplicate_buf_count = len(buf_dup[duplicate_sentence])

    ref_flag = False
    # 當有一方buf data數量為1時且另一方大於1時，先刪除buf data數量大於1的，才不用往回找
    # 因為如果刪除 <= 1的，刪除後該點的parent就無法使用，還必須要往回找
    if ref_buf_count > 1 and duplicate_buf_count <= 1:
        print("ref_buf_count > 1 and duplicate_buf_count <= 1")
        ref_flag = True
    elif ref_buf_count <= 1 and duplicate_buf_count > 1:
        print("ref_buf_count <= 1 and duplicate_buf_count > 1")
    # 如果兩個數量都大於1，刪除depth比較深的
    elif ref_buf_count > 1 and duplicate_buf_count > 1:
        print("ref_buf_count > 1 and duplicate_buf_count > 1")
        if Depth(ref_sentence, ref_concept) >= duplicate_depth:
            print("ref_depth > duplicate_depth")
            ref_flag = True
        else:
            print("duplicate_depth > ref_depth")
    else:
        print("ref_buf_count <= 1 and duplicate_buf_count <= 1:")
        ref_flag = True

    if ref_flag == True:
        print("根據ref_sentence去跳")
        return c1, ref_sentence, ref_concept
    print("根據duplicate_sentence去跳")
    return duplicate_concept_str, duplicate_sentence, duplicate_concept


def saveConcepts(concepts, c1, c2, c3, sentence):
    if c1:
        concepts[sentence][0] = c1  # search node
    if c2:
        concepts[sentence][1] = c2  # second node
    if c3:
        concepts[sentence][2] = c3  # third node

    if mode[sentence] == 2:
        concepts[sentence][2] = ""

    print("[", sentence, "][0]:", c1, "    [", sentence, "][1]:", c2, "    [", sentence, "][2]:", c3, sep="")
    print()


def printConcepts(*args):
    # print(args[0][0].__class__.__name__, "(printConcepts)")
    if len(args) == 1:
        # if args[0][0].__class__.__name__ == "Node":
        selected_node = args[0]
        print("Selected nodes: ")
        for i in range(len(selected_node)):
            sentence = currentSentence(i)
            current_concept = currentConcept(i)

            if (mode[sentence] != 2 and current_concept == 3) or (mode[sentence] == 2 and current_concept == 2) or (i == len(selected_node)-1):
                print(selected_node[i].data())
            else:
                print(selected_node[i].data(), "-", end="")

    # 一個template完整結束後，印出全部的concepts
    elif len(args) == 2:
        concepts = args[0]
        sentence = args[1]
        for s in range(sentence+1):
            print('-'.join(concepts[s]))

    # 印出某句的concepts
    elif len(args) == 3:
        c1 = args[0]
        c2 = args[1]
        c3 = args[2]
        print("C1:", c1, ", C2:", c2, ", C3:", c3, sep="")


def selectConceptFromDB(c1, c1_search_position, rel, buf, sentence=None):
    list_1 = list()
    list_2 = list()
    index_1 = list()
    index = list()

    if sentence is None:
        buf.clear()
    else:
        buf[sentence].clear()

    if c1_search_position == "Start":
        list_1 = CN_end
        list_2 = CN_start
    else:
        list_1 = CN_start
        list_2 = CN_end

    for i in range(len(CN_start)):
        if list_2[i] == c1:
            index_1.append(i)
    for i in range(len(index_1)):
        if CN_relation[index_1[i]] == rel:
            index.append(index_1[i])

    if sentence is None:
        for i in index:
            if c1 != list_1[i]:  # avoid duplication
                buf.append(list_1[i])
        return list(set(buf))
        # print(c1, "(", rel, ", ", c1_search_position, ")(", len(buf), "):", buf, sep="")

    for i in index:
        if c1 != list_1[i]:  # avoid duplication
            buf[sentence].append(list_1[i])
    buf[sentence] = list(set(buf[sentence]))
    # print(c1, "(", rel, ", ", c1_search_position, ")(", len(buf[sentence]), "):", buf[sentence], sep="")


# c1 means start concept(not search concept)
def selectSurfacetext(c1, cx, cx_search_position, rel):
    surfacetext = ""
    list_1 = list()
    list_2 = list()
    index_1 = list()
    index_2 = list()
    index = list()

    # list
    if cx_search_position == "Start":
        list_1 = CN_start
        list_2 = CN_end
    else:
        list_1 = CN_end
        list_2 = CN_start

    for i in range(len(CN_start)):
        if list_1[i] == cx:
            index_1.append(i)
    for i in range(len(index_1)):
        if list_2[index_1[i]] == c1:
            index_2.append(index_1[i])
    for i in range(len(index_2)):
        if CN_relation[index_2[i]] == rel:
            index.append(index_2[i])

    if not index:
        pdb.set_trace()
        sys.stderr.write("c1_search_position:"+invertStartEnd(cx_search_position)+"\ncx_search_position:"+cx_search_position+"\nC1:"+c1+"\nCx:"+cx+"\nRelation:"+rel+'\n')
    else:  # test之後要刪掉
        surfacetext = CN_surfacetext[index[0]]

    # print("surfacetext:", surfacetext)
    return surfacetext


# Initialize global variable
def templateInitialize():
    for i in range(MAX_SENTENCE):
        # numpy array
        mode[i] = -1
        backup_mode[i] = -1

        # list
        single_pos[i] = ""
        SurfaceText[i] = ""
        for j in range(2):
            template[i][j] = ""
            compound_pos[i][j] = ""
        for j in range(MAX_BACKUP):
            backup_template[i][j] = ""
        for j in range(3):
            assoc_flag[i][j] = ""
            sentiment[i][j] = ""


# 複製backup的值，並將backup_template往前遞補。
def updateTemplateAndMode(sentence):
    template[sentence][0] = backup_template[sentence][0]
    template[sentence][1] = backup_template[sentence][1]

    if template[sentence][0] == "":
        pdb.set_trace()
        print()

    mode[sentence] = backup_mode[sentence]
    backup_mode[sentence] = -1

    # for i in range(len(backup_template[sentence])):
        # print("backup_template[", sentence, "][", i, "]:", backup_template[sentence][i])

    for i in range(len(backup_template[sentence])-2):
        backup_template[sentence][i] = backup_template[sentence][i+2]
        backup_template[sentence][i+2] = ""

    # print("\nAfter updating:")
    # for i in range(len(backup_template[sentence])):
        # print("backup_template[", sentence, "]", i, "]:", backup_template[sentence][i])


def updateTemplate(sentence, rel, current_concept):
    backup_index = 0
    for backup_index in range(MAX_BACKUP):
        if backup_template[sentence][backup_index] == rel:
            break

    if backup_mode[sentence] == 2:
        template[sentence][0] = rel
        template[sentence][1] = ""
    elif current_concept == 2:
        template[sentence][0] = backup_template[sentence][backup_index]
        template[sentence][1] = backup_template[sentence][backup_index+1]
        backup_index += 1
    elif current_concept == 3:
        template[sentence][0] = backup_template[sentence][backup_index-1]
        template[sentence][1] = backup_template[sentence][backup_index]

    if template[sentence][0] == "":
        pdb.set_trace()
        print()

    # print("sentence:", sentence)
    # print("Before updating:")
    # print(backup_template[sentence])

    # 清除backup_index前面的資料(包含自己)
    for i in range(backup_index+1):
        backup_template[sentence][i] = ""
    # print(backup_template[sentence])
    # 將正在使用的backup_template後面的資料往前挪，原本的位置設為null
    insert_index = 0
    for i, rel in enumerate(backup_template[sentence][:]):
        if rel:
            backup_template[sentence].insert(insert_index, backup_template[sentence].pop(i))
            insert_index += 1

    # print("\nAfter updating:")
    # print(backup_template[sentence])


# change surfacetext when selecting a node.To make sure SurfaceText is the newest one
def updateSurfacetext(current_step, concepts, node, selected_node):
    print("----------updateSurfacetext-------------")
    # print("current_step:", current_step)

    sentence = currentSentence(current_step)
    current_concept = currentConcept(current_step)

    if (mode[sentence] == 2 and current_concept == 1) or (mode[sentence] != 2 and current_concept == 2):
        # print("\nBefore revising SurfaceText:" + SurfaceText[sentence])
        if mode[sentence] == 2 and current_concept == 1:
            c1 = node.data()
            c2 = selected_node.data()
            rel_1 = selected_node.relation()
            SurfaceText[sentence] = selectSurfacetext(c1, c2, selected_node.search_position(), rel_1)
        else:
            match = 0
            surfacetext_1 = ""
            surfacetext_2 = ""
            c1 = concepts[sentence][0]
            c2 = node.data()
            c3 = selected_node.data()
            rel_1 = node.relation()
            rel_2 = selected_node.relation()
            surfacetext_1 = selectSurfacetext(c1, c2, node.search_position(), rel_1)
            print("SurfaceText_1: " + surfacetext_1)
            if mode[sentence] == 1:
                c1, c2 = c2, c1
            surfacetext_2 = selectSurfacetext(c1, c3, selected_node.search_position(), rel_2)
            print("SurfaceText_2: " + surfacetext_2)

            # 1 means Start, 0 means End
            c1_search_position_to_c2 = 0
            c1_search_position_to_c3 = 0
            if node.search_position() == "End":  # c1_to_c2的search position為End
                c1_search_position_to_c2 = 1
            if selected_node.search_position() == "End":
                c1_search_position_to_c3 = 1

            if mode[sentence] == 1:
                c1_search_position_to_c2 = 1 if c1_search_position_to_c2 == 0 else 0
            if Convert(rel_1) > Convert(rel_2):
                c2, c3 = c3, c2
                rel_1, rel_2 = rel_2, rel_1
                surfacetext_1, surfacetext_2 = surfacetext_2, surfacetext_1
                c1_search_position_to_c2, c1_search_position_to_c3 = c1_search_position_to_c3, c1_search_position_to_c2
            match = c1_search_position_to_c2*2 + c1_search_position_to_c3  # match is binary presentation
            Generate(c1, c2, c3, rel_1, rel_2, surfacetext_1, surfacetext_2, sentence, match)
        # print("After revising SurfaceText:" + SurfaceText[sentence])


# Delete the template from the schema list and change schema
def changeSchema(schema):
    if len(schema) == 1:
        sys.stderr.write("Error(changeSchema). Please change the start concept\n")
        return True

    del schema[0]
    random.shuffle(schema)
    getattr(sys.modules[__name__], schema[0])()
    print("-------------Change to", schema[0], "-------------")
    return False


# Delete the useless concept in buf.It won't be selected in next round(例如第x句的c1引用了第三句的cy，便回到第三句重新選擇)
def deleteUselessConcept(removing_concept, ref_sentence, ref_concept, buf_c2, buf_c3):
    global c2_reselect_flag, c3_reselect_flag

    if ref_concept == 1:  # C1使用的是前面句子的C2 or end mode的C1
        print("將要刪除buf_c2[", ref_sentence, "](", len(buf_c2[ref_sentence]), ")的[", removing_concept, "]", sep="")
        if buf_c2[ref_sentence]:
            # test
            if removing_concept not in buf_c2[ref_sentence]:
                pdb.set_trace()
                print()
            # test
            buf_c2[ref_sentence].remove(removing_concept)
        else:
            pdb.set_trace()
            print()
        c2_reselect_flag = True
        c3_reselect_flag = False
    elif ref_concept == 2:  # C1使用的是前面句子的C3
        print("將要刪除buf_c3[", ref_sentence, "](", len(buf_c3[ref_sentence]), ")的[", removing_concept, "]", sep="")
        if buf_c3[ref_sentence]:
            # test
            if removing_concept not in buf_c3[ref_sentence]:
                pdb.set_trace()
                print()
            # test
            buf_c3[ref_sentence].remove(removing_concept)
        else:
            pdb.set_trace()
            print()
        c2_reselect_flag = False
        c3_reselect_flag = True
    # test
    else:
        print("ERROR!!!!")
        print("ref_concept:", ref_concept)
        pdb.set_trace()
        print()
    # test


# Delete the node(or node on upper layer) which is useless in tree
def deleteUselessNode(node, depth):
    sentence = currentSentence(depth)
    current_node = currentConcept(depth)
    sentence_backup = sentence

    root_sentence = node.getRootSentence(depth)
    root = node.getRoot()
    print("root:第", root_sentence, "句的[", root.data(), "]", sep="")
    print("將要刪除第", sentence, "句的C", current_node, ":["+node.data(), "](depth:", depth, ")", sep="")

    if not node.parent():
        sys.stderr.write("Error(deleteUselessNode). Can't delete root!!\n")
        return -100

    # 要刪除的是C1才要移動到C1參考的node；刪除C2或C3則不用往上移動
    if current_node == 1:
        ref_sentence = int(assoc_flag[sentence][0][0])
        ref_concept = int(assoc_flag[sentence][0][1])

        ref_concept = resetConceptsPosition(ref_sentence, ref_concept)
        ref_depth = Depth(ref_sentence, ref_concept)

        print("參考的depth:", ref_depth)
        while depth - ref_depth != 0:
            if node.parent():
                print("目前第", depth, "層, C", currentConcept(depth), ":", node.data(), ", parent C", currentConcept(depth-1), ":", node.parent().data(), sep="")
                node = node.parent()
                depth -= 1
            else:
                # pdb.set_trace()
                sys.stderr.write("Error(deleteUselessNode), can't delete root\n")
                return -100

    # node為root_sentence的c2或c3，或root_sentence是end mode的c2，刪除後只剩root一點，需要換schema
    sentence = currentSentence(depth)
    # node不為root_sentence end mode時的C3。刪除後只剩root一點，需要換schema
    if not node.parent() or (node.siblingSize() == 0) and (sentence == root_sentence) and not (mode[root_sentence] == 1 and depth == startDepth(root_sentence)+2):
        sys.stderr.write("Error(deleteUselessNode). There's only one root after deleting "+node.data()+"("+str(depth)+"). Change schema!!\n")
        return -100

    if node.siblingSize() == 0:  # 如果沒有sibling，代表刪除後parent也沒有children了
        print("[", node.data(), "](depth:", depth, ") has no sibling", sep="")
        if not backup_template[sentence_backup][0]:  # 有備用template時，不刪除parent node
            if currentConcept(depth) == 3 and mode[sentence] != 1:  # 非end mode且找不到C3時，代表該句的C1要刪除了，所以直接跳到C1
                delete_status = deleteUselessNode(node.parent().parent(), depth-2)
            else:
                delete_status = deleteUselessNode(node.parent(), depth-1)  # recursive call
            if delete_status == -100:
                return -100
    print("刪除[", node.data(), "](depth:", depth, ")", sep="")
    node.delete()
    return 0


# 順序為Tree的順序，沒有交換過
def printFinalResult(selected_node, schema):
    total_sentence = templateSentenceNum()
    concepts = [["" for x in range(3)] for y in range(MAX_SENTENCE)]

    print("\nFinal Result(progress):")
    for sentence in range(total_sentence):
        SurfaceText[sentence] = ""
        depth = startDepth(sentence)  # return start step index according to different sentence number
        c1 = selected_node[depth].data()
        c2 = selected_node[depth+1].data()
        concepts[sentence][0] = c1
        concepts[sentence][1] = c2
        # rel要更改，不是每一個要的都是template[sentence][0]，有可能因為備用的關係而更改了
        rel_1 = selected_node[depth+1].relation()
        pos_1 = selected_node[depth+1].search_position()
        surfacetext_c1_c2 = selectSurfacetext(c1, c2, pos_1, rel_1)
        # c1, c1_search_position, cx, rel)
        if mode[sentence] == 2:
            SurfaceText[sentence] = surfacetext_c1_c2
            print("Single sentence:", SurfaceText[sentence])
        else:
            print("Surfacetext_c1_c2:", surfacetext_c1_c2)
            c3 = selected_node[depth+2].data()
            concepts[sentence][2] = c3

            if mode[sentence] == 1:
                c1, c2 = c2, c1

            rel_2 = selected_node[depth+2].relation()
            pos_1 = selected_node[depth+2].search_position()

            surfacetext_c1_c3 = selectSurfacetext(c1, c3, pos_1, rel_2)
            print("Surfacetext_c1_c3:", surfacetext_c1_c3)

            # 1:Start, 0:End
            c1_search_position_to_c2 = 0
            c1_search_position_to_c3 = 0
            if selected_node[depth+1].search_position() == "End":  # c1_to_c2的search position為Start
                c1_search_position_to_c2 = 1
            if selected_node[depth+2].search_position() == "End":
                c1_search_position_to_c3 = 1

            if mode[sentence] == 1:
                c1_search_position_to_c2 = 1 if c1_search_position_to_c2 == 0 else 0
            if Convert(rel_1) > Convert(rel_2):
                c2, c3 = c3, c2
                rel_1, rel_2 = rel_2, rel_1
                surfacetext_c1_c2, surfacetext_c1_c3 = surfacetext_c1_c3, surfacetext_c1_c2
                c1_search_position_to_c2, c1_search_position_to_c3 = c1_search_position_to_c3, c1_search_position_to_c2
            saveConcepts(concepts, c1, c2, c3, sentence)
            match = c1_search_position_to_c2*2 + c1_search_position_to_c3   # match is binary presentation
            Generate(c1, c2, c3, rel_1, rel_2, surfacetext_c1_c2, surfacetext_c1_c3, sentence, match)

    print("\nFinal Result(Original):")
    for i in range(total_sentence):
        print(SurfaceText[i])

    replaceWithPronoun(concepts)
    print("\nFinal Result(replaced with pronouns)")
    for i in range(total_sentence):
        print(SurfaceText[i])

    concepts_backup = copy.deepcopy(concepts)
    """
    concept用同義詞取代，因為不是每個paragraph換完synonym後都還能保持原意，所以用neural network 做 prediction
    挑出前幾分數高的，從這裡面隨機挑選一個，當成final output
    """
    file_w.write('------------------------Synonym part------------------------\n')
    synonym_paragraph_score = dict()
    for i in range(MAX_SYNONYM_PARAGRAPH_NUM):
        pred_paragraph = SurfaceText[:total_sentence]
        concepts = copy.deepcopy(concepts_backup)
        if i != 0:  # 第一句留給原來的paragraph
            pred_paragraph = replaceWithSynonym(pred_paragraph, concepts)

        # paraphrae(送到predictScore和最後呈現的pred_graph是不一樣的)
        if schema[0] == "Template_2":
            supplement = Supplement_MotivatedByGoal(concepts[4][1])
            pred_paragraph[3] = pred_paragraph[3] + supplement + concepts[4][1]
            pred_paragraph.pop(4)
        elif schema[0] == "Template_3":
            pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
            # pred_paragraph[2] = pred_paragraph[2] + '，會讓人想要 ' + concepts[3][1]
            pred_paragraph.pop(3)
        elif schema[0] == "Template_4":
            if mode[3] == 2:
                supplement = Supplement_MotivatedByGoal(concepts[3][1])
                pred_paragraph[2] = pred_paragraph[2] + supplement + concepts[3][1]
            elif mode[3] == 1:
                pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][2], '').split())
            pred_paragraph.pop(3)
            pred_paragraph.pop(3)
        elif schema[0] == "Template_5":
            if mode[2] == 2:
                pred_paragraph[2] = "為了 " + concepts[2][1] + " 而 " + concepts[2][0] + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
            else:
                pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][0], '').split())
            pred_paragraph.pop(3)
            supplement = Supplement_MotivatedByGoal(concepts[5][1])
            pred_paragraph[3] = pred_paragraph[3] + supplement + concepts[5][1]
            pred_paragraph.pop(-1)
        elif schema[0] == "Template_7":
            supplement = Supplement_MotivatedByGoal(concepts[1][1])
            pred_paragraph[0] = pred_paragraph[0] + supplement + concepts[1][1]
            pred_paragraph.pop(1)
            pred_paragraph.pop(1)
            pred_paragraph.pop(3)
        elif schema[0] == "Template_8":
            if mode[3] == 2:
                supplement = Supplement_MotivatedByGoal(concepts[3][1])
                pred_paragraph[2] = pred_paragraph[2] + supplement + concepts[3][1]
            elif mode[3] == 1:
                pred_paragraph[2] = pred_paragraph[2] + '，' + ' '.join(pred_paragraph[3].replace(concepts[3][2], '').split())
            pred_paragraph.pop(3)
            pred_paragraph.pop(3)
        elif schema[0] == "Template_9":
            pred_paragraph.pop(2)
            pred_paragraph.pop(3)

        removeSurfacetextFormat(pred_paragraph)
        # remove brackets
        for j in range(len(pred_paragraph)):
            pred_paragraph[j] = pred_paragraph[j].replace('[', '')
            pred_paragraph[j] = pred_paragraph[j].replace(']', '')
            pred_paragraph[j] = ' '.join(pred_paragraph[j].split())

        score = predictScore(pred_paragraph)
        file_w.write('score:'+str(score)+'\n')
        if score in synonym_paragraph_score:
            synonym_paragraph_score[score].append(pred_paragraph + ['||'] + concepts)
        else:
            synonym_paragraph_score[score] = [pred_paragraph + ['||'] + concepts]

    synonym_paragraph_score = dict(sorted(synonym_paragraph_score.items(), key=lambda x:x[0], reverse=True))
    tmp_pred_paragraph = random.choice(random.choice(list( synonym_paragraph_score.values() )[:10]))  # 前10名隨機選一個

    for i, value in enumerate(tmp_pred_paragraph):
        if value == "||":
            pred_paragraph = tmp_pred_paragraph[:i]
            concepts = tmp_pred_paragraph[i+1:]
            break

    print("\nFinal Result(replaced with synonyms)")
    for sent in pred_paragraph:
        print(sent)
        file_w.write(sent+'\n')

    # 省略開頭concept
    if schema[0] == "Template_1":
        pred_paragraph[2] = re.sub(r'\b'+re.escape(concepts[2][0])+r'\b', '', pred_paragraph[2])
        pred_paragraph[2] = ' '.join(pred_paragraph[2].split())
        pred_paragraph[3] = re.sub(r'\b'+re.escape(concepts[3][0])+r'\b', '', pred_paragraph[3])
        pred_paragraph[3] = ' '.join(pred_paragraph[3].split())
    elif schema[0] == "Template_2":
        pred_paragraph[1] = re.sub(r'\b'+re.escape(concepts[1][0])+r'\b', '', pred_paragraph[1])
        pred_paragraph[1] = ' '.join(pred_paragraph[1].split())
        pred_paragraph[2] = re.sub(r'\b'+re.escape(concepts[2][0])+r'\b', '', pred_paragraph[2])
        pred_paragraph[2] = ' '.join(pred_paragraph[2].split())

    print("\nFinal Result:")
    for sent in pred_paragraph:
        print(sent)

    with open(str(GENERATED_PARAGRAPH_DIR_PATH)+'\\'+selected_node[0].data()+'_'+schema[0]+'.txt', 'a', encoding="UTF-8") as file_paragraph:
        file_paragraph.write(selected_node[0].data() + ', ' + schema[0] + '\n')
        print("\nschema:", schema[0])
        for sent in pred_paragraph:
            sent = ''.join(sent.split(' '))
            print(sent)
            file_paragraph.write(sent+'\n')
        file_paragraph.write('----------------------------\n')


# 目前只先針對每一個句子的第一個concept做替換
def replaceWithPronoun(concepts):
    cilin_indexes = list()
    total_sentence = templateSentenceNum()
    replace_flag = False

    for i in range(1, total_sentence):
        # for m in range(total_sentence):
        for m in range(i):
            for n in range(3):
                if concepts[m][n] == concepts[i][0]:
                    cilin_indexes = cilinIndexes(concepts[i][0])
                    for cilin_index in cilin_indexes:
                        if cilin_coding_sub1[cilin_index] == "A":     # A分類為人
                            SurfaceText[i] = SurfaceText[i].replace('['+concepts[i][0]+']', "[他]")
                            concepts[i][0] = "他"
                            replace_flag = True
                            break
                        elif cilin_coding_sub2[cilin_index] == "Bi":  # Bi分類為動物
                            SurfaceText[i] = SurfaceText[i].replace('['+concepts[i][0]+']', "[牠]")
                            concepts[i][0] = "牠"
                            replace_flag = True
                            break
                if replace_flag == True:
                    break
            if replace_flag == True:
                replace_flag = False
                break


def replaceWithSynonym(pred_paragraph, concepts):
    total_sentence = templateSentenceNum()
    for i in range(total_sentence):
        for j in range(3):
            if concepts[i][j]:
                if concepts[i][j] in synonym_dict:
                    # replaced_rate = 0.6
                    replaced_rate = 0.9
                else:
                    replaced_rate = 0

                if random.random() < replaced_rate:
                    synonyms = findSynonyms(concepts[i][j])
                    while synonyms:
                        synonym = random.choice(synonyms)
                        duplicate_depth, _ = duplicateCheck(synonym, concepts, i, j)
                        if duplicate_depth == -1:
                            pred_paragraph[i] = pred_paragraph[i].replace(concepts[i][j], synonym)
                            concepts[i][j] = synonym
                            break
                        else:
                            synonyms.pop(0)
    return pred_paragraph


# remove format and segmentation
def removeSurfacetextFormat(paragraph):
    for i, s in enumerate(paragraph):
        if s:
            # paragraph[i] = paragraph[i].replace('，', '')
            paragraph[i] = paragraph[i].replace('。', '')
            paragraph[i] = paragraph[i].replace('[[', '[')
            paragraph[i] = paragraph[i].replace(']]', ']')
            segs = jieba.lcut(paragraph[i], cut_all = False, HMM = False)
            sent = ' '.join(segs)
            paragraph[i] = ' '.join(sent.split())
            paragraph[i] = paragraph[i].replace('[ ', '[')
            paragraph[i] = paragraph[i].replace(' ]', ']')


"""
--  -> -
-+  -> -
-|  -> -
++  -> +
+-  -> +
+|  -> +
否+ -> -
否- -> +
否| -> |
"""
# Return sentiment polarity of the concept
def Sentiment(concept):
    # 否定詞
    privative = {"不", "不會", "不要", "不可", "不可以", "不能", "不能夠", "不得", "不怎麼", "不是", "不已", "不甘", "不及", "不行", "不必", "不用", "不須", "不需", "不再", "不常", "不大", "不曾", "絕不", "從不", "毫不", "從未", "毫無", "甭", "無", "無須", "無需", "無法", "絕不", "沒", "沒有", "免", "毋", "勿", "非", "弗", "莫", "未", "否", "忌", "禁止", "防止", "難以", "拒絕", "杜絕", "並非", "並不", "預防", "防止", "防範", "杜絕", "阻止", "避免","永不","決不"}

    polarity = ""
    concept_segs = concept2seg_dict[concept].split()

    for concept_seg in concept_segs:
        if concept_seg in privative:
            polarity = polarity + "否"
        elif concept_seg in positive:
            polarity = polarity + "+"
        elif concept_seg in negative:
            polarity = polarity + "-"
        elif concept_seg in neutral:
            polarity = polarity + "|"

    if not polarity:
        return "NotMatch"

    polarity = re.sub(r'\++', '+', polarity)
    polarity = re.sub(r'\|+', '|', polarity)
    polarity = polarity.replace("--", "-")
    polarity = polarity.replace("-+", "-")
    polarity = polarity.replace("-|", "-")
    polarity = polarity.replace("+-", "+")
    polarity = polarity.replace("+|", "+")
    polarity = polarity.replace("否+", "-")
    polarity = polarity.replace("否-", "+")
    polarity = polarity.replace("否|", "|")

    if "-" in polarity:
        return "Negative"
    if "+" in polarity:
        return "Positive"
    if "|" in polarity:
        return "Neutral"
    if "否" in polarity:
        return "Negative"


"""
sentiment[1][1] = "02"
sentiment[1][2] = "11"
"""
# return target sentiment('' or specific sentiment) of next layer
def targetSentiment(concepts, depth):
    sentence = currentSentence(depth)
    current_concept = currentConcept(depth)-1
    if current_concept == 0:
        return ""

    target_sentiment = "NotMatch"
    # e.g.: target_sentiment: "00", "Positive", ""
    while target_sentiment == "NotMatch":
        # specific sentiment
        target_sentiment = sentiment[sentence][current_concept]
        # have the same sentiment with previous concept
        if target_sentiment.isdigit():
            sentence, current_concept = int(target_sentiment[0]), int(target_sentiment[1])
            target_sentiment = Sentiment(concepts[sentence][current_concept])
            if target_sentiment == "NotMatch" and current_concept == 0:
                sentence, current_concept = int(assoc_flag[sentence][0][0]), int(assoc_flag[sentence][0][1])
    return target_sentiment


# Return concepts in list which sentiment is same as target_sentiment(target_sentiment: Positive, Negative, Neutral, NotMatch, Non-Negative, Non-Positive)
def sameSentiment(nodes_list, target_sentiment):
    target_sentiment_list = list()

    if target_sentiment == "NotMatch" or not target_sentiment:
        return nodes_list, 0
    if target_sentiment == "Non-Negative":
        target_sentiment_list.append("Positive")
        target_sentiment_list.append("Neutral")
    elif target_sentiment == "Non-Positive":
        target_sentiment_list.append("Negative")
        target_sentiment_list.append("Neutral")
    else:
        target_sentiment_list.append(target_sentiment)

    # target_sentiment為正or負時，後面接中性詞，盡量讓相反的詞性不被選到
    if target_sentiment == "Positive" or target_sentiment == "Negative":
        second_sentiment= "Neutral"
    else:
        second_sentiment= ""
    second_sentiment_list = list()

    same_sentiment_list = list()
    sentiment = ""
    for node in nodes_list:
        sentiment = Sentiment(node.data())
        for target_sentiment in target_sentiment_list:
            if sentiment == target_sentiment:
                same_sentiment_list.append(node)
                break
        if sentiment == second_sentiment:
            second_sentiment_list.append(node)

    same_sentiment_count = 0
    # 符合情緒的concept往前挪
    if same_sentiment_list:
        left_boundary = -1
        children_size = len(nodes_list)
        forward_step = int(round(children_size*SENTIMENT_FORWARD_STEP, 1))  #全部往前移動一定比例
        for i, node in enumerate(nodes_list[:]):
            if node in same_sentiment_list:
                nodes_list.remove(node)
                if i - forward_step <= left_boundary:
                    nodes_list.insert(left_boundary+1, node)
                    left_boundary += 1
                else:
                    nodes_list.insert(i-forward_step, node)
                same_sentiment_count += 1

    if second_sentiment:
        left_boundary = len(same_sentiment_list)
        for i, node in enumerate(nodes_list[:]):
            if node in second_sentiment_list:
                nodes_list.remove(node)
                nodes_list.insert(left_boundary, node)
                left_boundary += 1
                same_sentiment_count += 1
    # print("After sentiment check: ", nodes_list)
    return nodes_list, same_sentiment_count


# sort the concept_list in descending order according to relatedness to target_concept
def assocConcepts(target_concept, concept_list):
    other_wordsvec_list = list()
    OOV_index_set = set()

    # get target concept wordvec
    target_concept_wordvec = 0
    target_concept_segs = concept2seg_dict[target_concept].split()
    for concept_seg in target_concept_segs:
        if concept_seg in embedding_model:
            target_concept_wordvec += embedding_model[concept_seg]
    target_concept_wordvec /= len(target_concept_segs)

    concept_segs_list = list()
    for concept in concept_list:
        concept_segs_list.append(concept2seg_dict[concept])

    # get other concepts word vec
    for i, concept_segs in enumerate(concept_segs_list):
        words_vec = 0
        concept_segs = concept_segs.split()
        seg_num = len(concept_segs)
        for concept_seg in concept_segs:
            if concept_seg in embedding_model:
                words_vec += embedding_model[concept_seg]
        if isinstance(words_vec, np.ndarray) and seg_num != 0: # 2021
            words_vec /= seg_num  # average word embedding
            other_wordsvec_list.append(words_vec)
        else:
            other_wordsvec_list.append(np.ones(WE_DIMENSIONS))
            OOV_index_set.add(i)

    cosine_sim = embedding_model.cosine_similarities(target_concept_wordvec, other_wordsvec_list)

    # set cosine similarity score to 0 if the concept is OOV or score < 0
    related_concepts_dict = dict()
    for i, concept in enumerate(concept_list):
        if i in OOV_index_set or cosine_sim[i] < 0:
            cosine_sim[i] = 0
        related_concepts_dict[concept] = cosine_sim[i]

    # sort the scores in descending order
    related_concepts_dict = dict(sorted(related_concepts_dict.items(), key=lambda x:x[1], reverse=True))
    return related_concepts_dict


# concepts which are related to tartget_concept have higher probability to be simulated
def selectAssocConcept(target_concept, concept_list):
    related_concepts_dict = assocConcepts(target_concept, concept_list)

    # normalize the scores. the sum is 1
    relate_scores = list(related_concepts_dict.values())
    relate_scores_sum = sum(relate_scores)
    if relate_scores_sum == 0:
        sys.stderr.write("can't find associate concepts\n")
        return ''

    norm = [float(i)/relate_scores_sum for i in relate_scores]
    for i, concept in enumerate(related_concepts_dict.keys()):
        related_concepts_dict[concept] = norm[i]

    # find min in dict
    min_value = float('inf')
    for concept, score in related_concepts_dict.items():
        if score == 0:
            continue
        if score < min_value:
            min_value = score

    # natural log transformation(make right-skewed data(data concentrate on left side) more normal distribution)
    # 降低極值被選中的機率(相對提高其他值被選中的機率)
    for concept, score in related_concepts_dict.items():
        if score == 0:
            score = min_value/2
        related_concepts_dict[concept] = log(score)
    relate_scores = list(related_concepts_dict.values())

    # shift the data to positive
    for concept, score in related_concepts_dict.items():
        related_concepts_dict[concept] += abs(min(relate_scores))+0.1
    relate_scores = list(related_concepts_dict.values())

    # normalize the scores after log. the sum is 1
    relate_scores_sum = abs(sum(relate_scores))
    norm = [float(i)/relate_scores_sum for i in relate_scores]
    for i, concept in enumerate(related_concepts_dict.keys()):
        related_concepts_dict[concept] = norm[i]

    # select concept according to weights(scores)
    selected_concept = random.choices(list(related_concepts_dict.keys()), list(related_concepts_dict.values()))[0]
    return selected_concept


# Return the number of the template
def templateSentenceNum():
    total_sentence = 0
    while template[total_sentence][0]:
        total_sentence += 1
    return total_sentence


# 回復到該句使用時的位置
def resetConceptsPosition(*args):
    if len(args) == 2:
        ref_sentence = args[0]
        ref_concept = args[1]
        c1, c2, c3 = 0, 1, 2

        if mode[ref_sentence] != 2:
            rel_1 = template[ref_sentence][0]
            rel_2 = template[ref_sentence][1]
            if Convert(rel_1) > Convert(rel_2):
                c2, c3 = c3, c2
            if mode[ref_sentence] == 1:
                c1, c2 = c2, c1
            if ref_concept == c1:
                return 0
            if ref_concept == c2:
                return 1
            if ref_concept == c3:
                return 2
        else:
            return ref_concept

    elif len(args) == 3:
        concepts = args[0]
        sentence = args[1]
        current_concept = args[2]
        concepts_copy = copy.deepcopy(concepts)
        for i in range(sentence+1):
            if mode[i] != 2:
                if i != sentence:
                    rel_1 = template[i][0]
                    rel_2 = template[i][1]
                    if Convert(rel_1) > Convert(rel_2):
                        concepts_copy[i][1], concepts_copy[i][2] = concepts_copy[i][2], concepts_copy[i][1]
                if i != sentence or current_concept == 2:
                    if mode[i] == 1:
                        concepts_copy[i][0], concepts_copy[i][1] = concepts_copy[i][1], concepts_copy[i][0]
        return concepts_copy


# Return the total step(e.g., c1-c2-c3, total step=2)
def totalStep():
    total_sentence = templateSentenceNum()
    step = -1

    for i in range(total_sentence):
        if mode[i] == 2:
            step += 2
        else:
            step += 3
    # print("total_step: ", step)
    return step


# Return the start index of the step according to the current sentence
def startDepth(current_sentence):
    depth = 0

    for sentence in range(current_sentence):
        if mode[sentence] == 2:
            depth += 2
        else:
            depth += 3
    return depth


# tree的depth，都沒有交換過
def Depth(sentence, concept):
    return startDepth(sentence)+concept


# Return the start index of the sentence according to the depth
def currentSentence(depth):
    sentence = 0
    step = -1
    total_sentence = templateSentenceNum()

    for sentence in range(total_sentence):
        if mode[sentence] == 2:
            step += 2
        else:
            step += 3
        if step >= depth:
            break
    return sentence


# To decide current node is c1, c2 or c3 according to the depth. Return value: 1:C1, 2:C2, 3:C3
def currentConcept(depth):
    sentence = 0
    step = -1
    total_sentence = templateSentenceNum()

    for sentence in range(total_sentence):
        if mode[sentence] == 2:
            step += 2
            if step < depth:
                continue
            if step == depth:
                return 2
            if step-1 == depth:
                return 1
            return -1
        else:
            step += 3
            if step < depth:
                continue
            if step == depth:
                return 3
            if (step-1) == depth:
                return 2
            if step-2 == depth:
                return 1
            return -1


def invertStartEnd(s):
    if len(s) > 1:
        if s == "Start":
            return "End"
        return "Start"
    else:
        if s == '1':
            return '2'
        if s == '2':
            return '1'
        sys.stderr.write("ERROR!(invertStartEnd)\n")
        sys.exit(-1)


def swapConcepts(concepts):
    #轉成C1為search concept, C2為normal oerder relation_1所連接到的concept, C3為normal oerder relation_2所連接到的concept
    for i in range(templateSentenceNum()):
        if mode[i] == 2:
            continue
        # end mode swapping
        if mode[i] == 1 and concepts[i][1]:
            concepts[i][0], concepts[i][1] = concepts[i][1], concepts[i][0]
        # reverse relation swapping
        if Convert(template[i][0]) > Convert(template[i][1]) and concepts[i][1] and concepts[i][2]:
            concepts[i][1], concepts[i][2] = concepts[i][2], concepts[i][1]


# Return all cilin indexes with string s
def cilinIndexes(s):
    indexList = list()

    for i in range(len(cilin_data)):
        if s == cilin_data[i]:
            indexList.append(i)
    return indexList


"""
def non_outliers_modified_z_score(a):
    #MAD(median absolute deviation): median(abs(xi-median(X)))
    #modified z_score: 0.6745*(xi-median(X)) / MAD

    threshold = 3.5
    median_ = np.median(a)
    MAD = np.median([np.abs(a - median_)])
    modified_z_scores = 0.6745 * (a - median_) / (MAD if MAD else 1.)
    return a[np.abs(modified_z_scores) < threshold]
"""


# Load ConceptNet, Cilin, Sentiment DB
def loadDB():
    print("Load DB...")
    ConceptNet_DB_path = CURRENT_PATH / r"data\ConceptNet.db"

    conn = sqlite3.connect(ConceptNet_DB_path)
    c = conn.cursor()
    # load cilin DB
    results = c.execute("SELECT Data, Coding_sub1, Coding_sub2 FROM Cilin_revised").fetchall()
    for data, coding_sub1, coding_sub2 in results:
        cilin_data.append(data)
        cilin_coding_sub1.append(coding_sub1)
        cilin_coding_sub2.append(coding_sub2)

    # load ConceptNet DB
    results = c.execute(f"Select Start, End, Relation, SurfaceText FROM {DATABASE}").fetchall()
    for start, end, rel, surface in results:
        CN_start.append(start)
        CN_end.append(end)
        CN_relation.append(rel)
        CN_surfacetext.append(surface)

    # load ConceptNet DB(to select a start concept)
    results = c.execute(f"Select DISTINCT Start FROM {DATABASE} WHERE Relation='CapableOf' OR Relation='Desires' OR Relation='Have' OR (Relation='NotDesires' AND (SurfaceText LIKE '%] 厭惡%' OR SurfaceText LIKE '%] 痛恨%' OR SurfaceText LIKE '%] 懼怕%'))").fetchall()
    for start in results:
        CN_start_concept.append(start[0])

    # load Sentiment DB
    results = c.execute("SELECT Positive, Negative, Neutral FROM Sentiment").fetchall()
    for positive_, negative_, neutral_ in results:
        if positive_:
            positive.add(positive_)
        if negative_:
            negative.add(negative_)
        if neutral_:
            neutral.add(neutral_)

    conn.close()


def loadSynonyms(vocab_list_150):
    global synonym_dict

    vocab_set_150 = set(vocab_list_150)
    # cilin synonym
    cilin_path = CURRENT_PATH / r"data\Cilin_revised.txt"
    ConceptNet_DB_path = CURRENT_PATH / r"data\ConceptNet.db"
    cilin_word2categories_dict = dict()
    cilin_category2words_dict = dict()
    cilin_synonym_dict = dict()
    with open(cilin_path, 'r', encoding="UTF-8") as file_r:
        for line in file_r:
            _,word,_,_,_,_,_,_,star,category = line.rstrip().split(',')
            if word not in vocab_set_150 or len(word) <= 1 or star == "*":  # 單字的歧義太多，不使用。*代表的是relate concepts
                continue

            if word in cilin_word2categories_dict:
                cilin_word2categories_dict[word].append(category)
            else:
                cilin_word2categories_dict[word] = [category]

            if category in cilin_category2words_dict:
                cilin_category2words_dict[category].add(word)
            else:
                cilin_category2words_dict[category] = {word}

        # remove words with multiple categories
        for category, words in cilin_category2words_dict.copy().items():
            for word in words.copy():
                if len(cilin_word2categories_dict[word]) > 1:
                    cilin_category2words_dict[category].remove(word)
                    if not cilin_category2words_dict[category]:
                        cilin_category2words_dict.pop(category)

        # 合併為 cilin[word]:{synonyms}
        for word in cilin_word2categories_dict:
            if len(cilin_word2categories_dict[word]) == 1:  # 採用分類為一種的
                for category in cilin_word2categories_dict[word]:  # words may have different categories
                    if word in cilin_synonym_dict:
                        cilin_synonym_dict[word] = cilin_synonym_dict[word].union(cilin_category2words_dict[category])
                    else:
                        cilin_synonym_dict[word] = copy.deepcopy(cilin_category2words_dict[category])
                    # print(word,cilin_synonym_dict[word])
                    cilin_synonym_dict[word].remove(word)
                    if not cilin_synonym_dict[word]:
                        del cilin_synonym_dict[word]
        del cilin_word2categories_dict
        del cilin_category2words_dict

    # ConceptNet synonym
    conn = sqlite3.connect(ConceptNet_DB_path)
    c = conn.cursor()
    results = c.execute(f"Select DISTINCT Start, End FROM {DATABASE} WHERE Relation = 'Synonym'").fetchall()
    conn.close()
    conceptnet_synonym_dict = dict()
    for start_concept, end_concept in results:
        if len(start_concept) <= 1 or len(end_concept) <= 1:
            continue
        if start_concept not in vocab_set_150 or end_concept not in vocab_set_150:
            continue

        if start_concept in conceptnet_synonym_dict:
            conceptnet_synonym_dict[start_concept].add(end_concept)
        else:
            conceptnet_synonym_dict[start_concept] = {end_concept}
        if end_concept in conceptnet_synonym_dict:
            conceptnet_synonym_dict[end_concept].add(start_concept)
        else:
            conceptnet_synonym_dict[end_concept] = {start_concept}

    # union the synonym words of Cilin and ConceptNet
    for concept in conceptnet_synonym_dict:
        if concept in cilin_synonym_dict:
            cilin_synonym_dict[concept] = cilin_synonym_dict[concept].union(conceptnet_synonym_dict[concept])
        else:
            cilin_synonym_dict[concept] = conceptnet_synonym_dict[concept]
    synonym_dict = copy.deepcopy(cilin_synonym_dict)
    del cilin_synonym_dict
    del conceptnet_synonym_dict

    print("Synonyms size:", len(synonym_dict))


def findSynonyms(concept):
    # single word may contain more ambiguities
    if len(concept) == 1 or concept not in synonym_dict:
        return []

    synonyms = list(synonym_dict[concept])
    concept_len = len(concept)
    for synonym in synonyms[:]:
        if concept_len > len(synonym)+2 or concept_len < len(synonym)-2:
            synonyms.remove(synonym)
    return synonyms


def load_obj(file_path):
    with open(file_path, 'rb') as file_r:
        return pickle.load(file_r)


def Supplement_MotivatedByGoal(concept):
    segs = concept2seg_dict[concept].split()
    pos = conceptnet_pos_combined_dict[' '.join(segs)]
    pos = POS(pos)

    if pos == "V":
        supplement = "來"
    elif pos == "N" or pos == "unknown":
        supplement = "，是為了"
    return supplement


def POS(pos):
    if "V" in pos:
        return "V"
    if "N" in pos:
        return "N"
    return "unknown"


""" Template """
""" assoc_flag[][x]:當x不為0時做associate checking """
def Template_1():
    print("--------------------Schema 1--------------------")

    template[0][0] = "AtLocation"
    template[0][1] = "HasProperty"
    template[1][0] = "CapableOf"
    template[1][1] = "CapableOf"
    template[2][0] = "Desires"
    template[2][1] = "Desires"
    template[3][0] = "IsA"
    template[3][1] = "Causes"
    backup_template[3][0] = "IsA"
    backup_template[3][1] = "CausesDesire"

    mode[0] = 0
    mode[1] = 0
    mode[2] = 0
    mode[3] = 0
    backup_mode[3] = 0

    assoc_flag[1][0] = "00"
    assoc_flag[2][0] = "00"
    assoc_flag[3][0] = "00"

    sentiment[1][1] = "02"
    sentiment[1][2] = "02"
    sentiment[2][1] = "02"
    sentiment[2][2] = "02"
    sentiment[3][1] = "02"
    sentiment[3][2] = "02"

def Template_2():
    print("--------------------Schema 2--------------------")

    template[0][0] = "AtLocation"
    template[0][1] = "HasProperty"
    template[1][0] = "Causes"
    template[1][1] = "IsA"
    template[2][0] = "CausesDesire"
    template[2][1] = "CausesDesire"
    template[3][0] = "HasSubevent"
    template[3][1] = "HasProperty"
    template[4][0] = "MotivatedByGoal"  # 放到第四句後面

    mode[0] = 0
    mode[1] = 0
    mode[2] = 0
    mode[3] = 1
    mode[4] = 2

    assoc_flag[1][0] = "00"
    assoc_flag[2][0] = "00"
    assoc_flag[3][0] = "21"
    assoc_flag[4][0] = "30"

    assoc_flag[1][1] = "02"
    assoc_flag[2][1] = "02"
    assoc_flag[3][1] = "00"
    assoc_flag[4][1] = "21"

    compound_pos[3][1] = "2"

    single_pos[4] = "Start"

    sentiment[1][1] = "02"
    sentiment[1][2] = "02"
    sentiment[2][1] = "02"
    sentiment[2][2] = "02"
    sentiment[3][1] = "02"
    sentiment[3][2] = "02"


#暫時不採用
"""
def Template_3():
    print("--------------------Schema 3--------------------")

    template[0][0] = "Desires"
    template[0][1] = "IsA"
    template[1][0] = "HasFirstSubevent"
    template[1][1] = "MotivatedByGoal"
    template[2][0] = "Causes"
    template[2][1] = "Causes"
    template[3][0] = "CausesDesire"
    template[4][0] = "NotDesires"
    template[4][1] = "HasProperty"

    mode[0] = 0
    mode[1] = 1
    mode[2] = 0
    mode[3] = 2
    mode[4] = 1

    assoc_flag[1][0] = "01"
    assoc_flag[2][0] = "01"
    assoc_flag[3][0] = "22"
    assoc_flag[4][0] = "00"

    assoc_flag[1][2] = "00"
    assoc_flag[2][1] = "00"
    assoc_flag[2][2] = "21"
    assoc_flag[3][1] = "00"

    compound_pos[1][0] = "2"
    compound_pos[1][1] = '1'

    single_pos[3] = "Start"

    sentiment[4][2] = "Negative"
"""

def Template_4():
    print("--------------------Schema 4--------------------")

    # Initialize relation template
    template[0][0] = "CapableOf"
    template[0][1] = "MotivatedByGoal"
    template[1][0] = "CausesDesire"
    template[1][1] = "HasProperty"
    template[2][0] = "HasSubevent"
    template[3][0] = "MotivatedByGoal"
    template[3][1] = "HasProperty"
    backup_template[3][0] = "MotivatedByGoal"
    template[4][0] = "Causes"  # 隱藏
    template[5][0] = "HasProperty"
    template[5][1] = "Causes"
    backup_template[5][0] = "HasProperty"
    backup_template[5][1] = "CausesDesire"

    # Initialize mode
    mode[0] = 1
    mode[1] = 0
    mode[2] = 2
    mode[3] = 1
    backup_mode[3] = 2
    mode[4] = 2
    mode[5] = 0
    backup_mode[5] = 0

    # Initialize association flag
    assoc_flag[1][0] = "00"
    assoc_flag[2][0] = "02"
    assoc_flag[3][0] = "21"
    assoc_flag[4][0] = "21"
    assoc_flag[5][0] = "41"

    # associate with previous concept
    assoc_flag[0][2] = "01"
    assoc_flag[1][1] = "01"
    assoc_flag[1][2] = "11"
    assoc_flag[2][1] = "01"
    assoc_flag[3][1] = "02"

    # 如果compound relation的其中一個search position為both(Start或End都可以)，在不同的template裡使用情況也各有不同，所以根據不同情況，自行給定search_position
    # [x][y] = "z"
    # x:sentence, y: 0:C1-C2, 1:C1-C3, z: 1:Start, 2:End
    compound_pos[3][1] = "2"

    # single sentence search place
    single_pos[2] = "Start"
    single_pos[3] = "Start"
    single_pos[4] = "Start"

    #雖然sentiment[sent][0]用不到，但為了保持和concepts的一致性所以空出來
    sentiment[1][1] = "02"
    sentiment[1][2] = "11"

    sentiment[2][2] = "02"
    sentiment[3][1] = "21"
    sentiment[3][2] = "21"

    sentiment[4][1] = "21"

    sentiment[5][1] = "41"
    sentiment[5][2] = "41"

# 5,6,7,8,9 的assoc_flag及情緒未檢查完成
def Template_5():
    """最後呈現時，把最後一句放到倒數第二句後面"""
    print("--------------------Schema 5--------------------")

    template[0][0] = "AtLocation"
    template[0][1] = "CapableOf"
    template[1][0] = "HasFirstSubevent"
    template[1][1] = "HasSubevent"
    template[2][0] = "HasProperty"
    template[2][1] = "MotivatedByGoal"
    backup_template[2][0] = "MotivatedByGoal"
    template[3][0] = "Causes"           # 合併到第3句
    template[4][0] = "Causes"
    template[4][1] = "CausesDesire"
    template[5][0] = "MotivatedByGoal"  # 合併到第4句

    mode[0] = 0
    mode[1] = 0
    mode[2] = 0
    backup_mode[2] = 2
    mode[3] = 2
    mode[4] = 1
    mode[5] = 2

    assoc_flag[1][0] = "02"
    assoc_flag[2][0] = "11"
    assoc_flag[3][0] = "11"
    assoc_flag[4][0] = "02"
    assoc_flag[5][0] = "42"

    assoc_flag[0][2] = "01"
    assoc_flag[1][1] = "00"
    assoc_flag[1][2] = "00"

    compound_pos[1][1] = "1"
    compound_pos[2][1] = "1"

    single_pos[2] = "Start"
    single_pos[3] = "Start"
    single_pos[5] = "Start"

    # sentiment[3][0] = "Non-Negative"
    sentiment[3][1] = "Positive"


def Template_6():
    print("--------------------Schema 6--------------------")

    template[0][0] = "Desires"
    template[1][0] = "Causes"
    template[1][1] = "HasProperty"
    template[2][0] = "NotDesires"
    template[2][1] = "Causes"
    backup_template[2][0] = "NotDesires"
    backup_template[2][1] = "HasProperty"
    template[3][0] = "CapableOf"
    template[3][1] = "HasSubevent"
    backup_template[3][0] = "CapableOf"
    backup_template[3][1] = "Causes"

    mode[0] = 2
    mode[1] = 0
    mode[2] = 1
    backup_mode[2] = 1
    mode[3] = 1
    backup_mode[3] = 1

    assoc_flag[1][0] = "01"
    assoc_flag[2][0] = "00"
    assoc_flag[3][0] = "20"

    assoc_flag[2][1] = "01"  #5.24

    compound_pos[2][0] = "1"
    compound_pos[3][1] = "1"

    single_pos[0] = "Start"

    sentiment[0][1] = "Positive"
    sentiment[1][1] = "Positive"
    # sentiment[1][1] = "Non-Negative"
    sentiment[1][2] = "Positive"
    sentiment[2][2] = "Negative"
    sentiment[3][1] = "Negative"
    sentiment[3][2] = "Negative"


def Template_7():
    print("--------------------Schema 7--------------------")

    template[0][0] = "CapableOf"
    template[0][1] = "HasFirstSubevent"
    template[1][0] = "MotivatedByGoal"     # 合併到第1句
    template[2][0] = "HasSubevent"         # 隱藏
    template[3][0] = "HasProperty"
    template[3][1] = "MotivatedByGoal"
    template[4][0] = "NotDesires"
    template[4][1] = "Causes"
    template[5][0] = "Causes"              # 隱藏
    template[6][0] = "HasSubevent"
    template[6][1] = "MotivatedByGoal"
    backup_template[6][0] = "CausesDesires"
    backup_template[6][1] = "MotivatedByGoal"

    mode[0] = 1
    mode[1] = 2
    mode[2] = 2
    mode[3] = 0
    mode[4] = 1
    mode[5] = 2
    mode[6] = 1
    backup_mode[6] = 1

    assoc_flag[1][0] = "02"
    assoc_flag[2][0] = "00"
    assoc_flag[3][0] = "21"
    assoc_flag[4][0] = "21"
    assoc_flag[5][0] = "21"
    assoc_flag[6][0] = "51"

    assoc_flag[1][1] = "01"
    assoc_flag[2][1] = "01"

    compound_pos[3][1] = '1'
    compound_pos[4][0] = '1'
    compound_pos[6][0] = '2'

    single_pos[1] = "Start"
    single_pos[2] = "Start"
    single_pos[5] = "Start"

    sentiment[4][2] = "Negative"
    sentiment[5][1] = "Negative"


def Template_8():
    print("--------------------Schema 8--------------------")

    template[0][0] = "CapableOf"
    template[0][1] = "NotDesires"
    template[1][0] = "Causes"
    backup_template[1][0] = "CausesDesire"
    template[2][0] = "Causes"
    template[2][1] = "HasSubevent"
    backup_template[2][0] = "Causes"
    backup_template[2][1] = "CausesDesire"
    template[3][0] = "MotivatedByGoal"     # 接到上一句
    template[3][1] = "HasProperty"
    backup_template[3][0] = "MotivatedyGoal"
    template[4][0] = "Causes"              # 隱藏
    template[5][0] = "Causes"
    template[5][1] = "HasProperty"
    backup_template[5][0] = "CausesDesire"
    backup_template[5][1] = "HasProperty"

    mode[0] = 1
    mode[1] = 2
    backup_mode[1] = 2
    mode[2] = 1
    backup_mode[2] = 1
    mode[3] = 1
    backup_mode[3] = 2
    mode[4] = 2
    mode[5] = 0
    backup_mode[5] = 0

    assoc_flag[1][0] = "02"
    assoc_flag[2][0] = "00"
    assoc_flag[3][0] = "22"
    assoc_flag[4][0] = "22"
    assoc_flag[5][0] = "41"

    assoc_flag[2][2] = "01"
    assoc_flag[5][1] = "01"

    compound_pos[0][0] = "2"
    compound_pos[0][1] = "1"
    compound_pos[2][0] = "2"
    compound_pos[2][1] = "1"
    compound_pos[3][1] = "2"

    single_pos[1] = "Start"
    single_pos[3] = "Start"
    single_pos[4] = "Start"

    sentiment[0][2] = "Negative"
    sentiment[3][2] = "Positive"
    sentiment[4][1] = "Negative"
    sentiment[5][1] = "Negative"
    sentiment[5][2] = "Negative"

# """
# 測試duplicate
def Template_9():
    print("--------------------Schema 9--------------------")
    template[0][0] = "CapableOf"
    template[0][1] = "HasProperty"
    template[1][0] = "HasFirstSubevent"
    template[1][1] = "HasSubevent"
    template[2][0] = "HasSubevent"      # 隱藏
    template[3][0] = "NotDesires"
    template[3][1] = "Causes"
    template[4][0] = "MotivatedByGoal"  # 隱藏
    template[5][0] = "Causes"

    mode[0] = 0
    mode[1] = 0
    mode[2] = 2
    mode[3] = 1
    mode[4] = 2
    mode[5] = 2

    assoc_flag[1][0] = "01"
    assoc_flag[2][0] = "01"
    assoc_flag[3][0] = "21"
    assoc_flag[4][0] = "01"
    assoc_flag[5][0] = "41"

    compound_pos[1][1] = "1"
    compound_pos[3][0] = "1"

    single_pos[2] = "Start"
    single_pos[4] = "Start"
    single_pos[5] = "Start"

# For surfacetext_1
def Supplement_rel_1(s):
    trans = ""

    # AtLocation
    if "] 裡" in s:
        trans = "裡"
    elif "] 外" in s:
        rand = random.randint(0, 2)
        if rand == 0:
            trans = "外"
        else:
            trans = "外面"
    elif "] 上" in s:
        rand = random.randint(0, 2)
        if rand == 0:
            trans = "上面"
        else:
            trans = "上方"
    elif "] 下" in s:
        rand = random.randint(0, 3)
        if rand == 0:
            trans = "下"
        elif rand == 1:
            trans = "下面"
        else:
            trans = "下方"

    # Causes
    if "] 會帶來 [" in s:
        trans = "帶來"
    elif "] 會引發 [" in s:
        trans = "引發"
    elif "] 會令人 [" in s or "] 所以 [" in s:
        trans = "令人"

    # CausesDesire
    if "] 會令人想要 [" in s:
        trans = "會令人想要"
    elif "] 的時候會想要 [" in s:
        trans = "時會想要"

    # Desires
    rand = random.randint(0, 4)
    if "] 喜歡 [" in s:
        if rand == 0:
            trans = "喜歡"
        elif rand == 1:
            trans = "喜愛"
        elif rand == 2:
            trans = "熱愛"
        elif rand == 3:
            trans = "愛好"
        else:
            trans = "喜好"
    elif "] 想要 [" in s:
        trans = "想要"

    # NotDesires
    if "厭惡 [" in s:
        rand = random.randint(0, 4)
        if rand == 0:
            trans = "厭惡"
        elif rand == 1:
            trans = "討厭"
        elif rand == 2:
            trans = "嫌惡"
        elif rand == 3:
            trans = "痛惡"
        else:
            trans = "厭煩"
    elif "懼怕 [" in s:
        rand = random.randint(0, 3)
        if rand == 0:
            trans = "懼怕"
        elif rand == 1:
            trans = "畏懼"
        elif rand == 2:
            trans = "生怕"
        else:
            trans = "害怕"
    elif "痛恨 [" in s:
        rand = random.randint(0, 1)
        if rand == 0:
            trans = "痛恨"
        else:
            trans = "憎恨"

    #MadeOf
    if "] 製成" in s:
        trans = "製成"
    elif "] 組成" in s:
        trans = "組成"
    elif "] 可以做成 [" in s:
        trans = "做成"
    elif "] 的原料是 [" in s:
        rand = random.randint(0, 2)
        if rand == 0:
            trans = "做成"
        elif rand == 1:
            trans = "組成"
        else:
            trans = "製成"
    return trans


# For surfacetext_2
def Supplement_rel_2(s):
    trans = ""

    # Causes
    if "] 會帶來 [" in s:
        trans = "帶來"
    elif "] 會引發 [" in s:
        trans = "引發"
    elif "] 會令人 [" in s or "] 所以 [" in s:
        trans = "令人"

    # CausesDesire
    if "] 會令人想要 [" in s:
        trans = "會令人想要"
    elif "] 的時候會想要 [" in s:
        trans = "時會想要"

    # Desires
    rand = random.randint(0, 4)
    if "喜歡 [" in s:
        if rand == 0:
            trans = "喜歡"
        elif rand == 1:
            trans = "喜愛"
        elif rand == 2:
            trans = "熱愛"
        elif rand == 3:
            trans = "愛好"
        else:
            trans = "喜好"
    elif "] 想要 [" in s:
        trans = "想要"

    # NotDesires
    if "厭惡 [" in s:
        rand = random.randint(0, 4)
        if rand == 0:
            trans = "厭惡"
        elif rand == 1:
            trans = "討厭"
        elif rand == 2:
            trans = "嫌惡"
        elif rand == 3:
            trans = "痛惡"
        else:
            trans = "厭煩"
    elif "懼怕 [" in s:
        rand = random.randint(0, 3)
        if rand == 0:
            trans = "懼怕"
        elif rand == 1:
            trans = "畏懼"
        elif rand == 2:
            trans = "生怕"
        else:
            trans = "害怕"
    elif "] 痛恨 [" in s:
        rand = random.randint(0, 1)
        if rand == 0:
            trans = "痛恨"
        else:
            trans = "憎恨"

    # MadeOf
    if "] 製成" in s:
        trans = "製成"
    elif "] 組成" in s:
        trans = "組成"
    elif "] 可以做成 [" in s:
        trans = "做成"
    elif "] 的原料是 [" in s:
        rand = random.randint(0, 2)
        if rand == 0:
            trans = "做成"
        elif rand == 1:
            trans = "組成"
        else:
            trans = "製成"
    return trans


# Convert relations to number
def Convert(s):
    if s == "AtLocation":
        num = 0
    elif s == "CapableOf":
        num = 1
    elif s == "Causes":
        num = 2
    elif s == "CausesDesire":
        num = 3
    elif s == "Desires":
        num = 4
    elif s == "NotDesires":
        num = 5
    elif s == "HasFirstSubevent":
        num = 6
    elif s == "HasProperty":
        num = 7
    elif s == "HasSubevent":
        num = 8
    elif s == "IsA":
        num = 9
    elif s == "MadeOf":
        num = 10
    elif s == "MotivatedByGoal":
        num = 11
    elif s == "PartOf":
        num = 12
    elif s == "SymbolOf":
        num = 13
    elif s == "MayUse":
        num = 14
    elif s == "Have":
        num = 15
    else:
        num = -1
    return num


# Return value represents the final search combination，0:(end, end), 1:(end, start), 2:(start, end), 3:(start, start)
def Match(match_str, swap_flag, sentence):
    c1_search_position_to_c2 = match_str[0]  # left digit
    c1_search_position_to_c3 = match_str[1]  # right digit
    left_num = right_num = 0

    if c1_search_position_to_c2 == '3':
        c1_search_position_to_c2 = compound_pos[sentence][0]
    if c1_search_position_to_c3 == '3':
        c1_search_position_to_c3 = compound_pos[sentence][1]

    if swap_flag:  # because the relation order is opposite
        c1_search_position_to_c2, c1_search_position_to_c3 = c1_search_position_to_c3, c1_search_position_to_c2

    if match_str == "-1":
        sys.stderr.write("ERROR!!!!!!沒有這個組合!!!!!\n")
        return -1

    # means C1 is in start position, and C2 is in End position
    if c1_search_position_to_c2 == '1':
        left_num = 2
    if c1_search_position_to_c3 == '1':
        right_num = 1

    # print("match:", (left_num + right_num))
    return left_num + right_num


# Generate sentence
def Generate(c1, c2, c3, rel_1, rel_2, surfacetext_1, surfacetext_2, sentence, match):
    trans_1 = trans_2 = ""
    match_status = True

    trans_1 = Supplement_rel_1(surfacetext_1)
    trans_2 = Supplement_rel_2(surfacetext_2)

    # 0
    if rel_1 == "AtLocation":
        if rel_2 == "AtLocation":
            rand = random.randint(0, 1)
            if match == 0:
                if rand == 0:
                    SurfaceText[sentence] = "可以在[" + c1 + "]" + trans_1 + "找到[" + c2 + "]和[" + c3 + "]"
                else:
                    SurfaceText[sentence] = "[" + c1 + "]" + trans_1 + "有[" + c2 + "]和[" + c3 + "]"
            elif match == 3:
                if rand == 0:
                    SurfaceText[sentence] = "可以在[" + c2 + "]和[" + c3 + "]" + trans_2 + "找到[" + c1 + "]"
                else:
                    SurfaceText[sentence] = "[" + c2 + "]和[" + c3 + "]" + trans_2 + "都有[" + c1 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "CapableOf":
            rand = random.randint(0, 3)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有會[" + c3 + "]的[" + c1 + "]"
            elif rand == 1:
                SurfaceText[sentence] = "可以在[" + c2 + "]" + trans_1 + "找到會[" + c3 + "]的[" + c1 + "]"
            elif rand == 2:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]會[" + c3 + "]"
            elif rand == 3:
                SurfaceText[sentence] = "[" + c1 + "]在[" + c2 + "]" + trans_1 + "[" + c3 + "]"

        elif rel_2 == "Causes":
            rand = random.randint(0, 2)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有會" + trans_2 + "[" + c3 + "]的[" + c1 + "]"
            elif rand == 1:
                SurfaceText[sentence] = "可以在[" + c2 + "]" + trans_1 + "找到會" + trans_2 + "[" + c3 + "]的[" + c1 + "]"
            elif rand == 2:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]會" + trans_2 + "[" + c3 + "]"

        elif rel_2 == "CausesDesire":
            rand = random.randint(0, 2)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有會令人想要[" + c3 + "]的[" + c1 + "]"
            elif rand == 1:
                SurfaceText[sentence] = "可以在[" + c2 + "]" + trans_1 + "找到會令人想要[" + c1 + "]的[" + c2 + "]"
            elif rand == 2:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]會令人想要[" + c3 + "]"

        elif rel_2 == "Desires":
            rand = random.randint(0, 2)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]" + trans_2 + "[" + c2 + "]"
            elif rand == 1:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有" + trans_2 + "[" + c3 + "]的[" + c1 + "]"
            elif rand == 2:
                SurfaceText[sentence] = "可以在[" + c2 + "]" + trans_1 + "找到" + trans_2 + "[" + c3 + "]的[" + c1 + "]"

        elif rel_2 == "NotDesires":
            rand = random.randint(0, 2)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]" + trans_2 + "[" + c3 + "]"
            elif rand == 1:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有" + trans_2 + "[" + c3 + "]的[" + c1 + "]"
            elif rand == 2:
                SurfaceText[sentence] = "可以在[" + c2 + "]" + trans_1 + "找到" + trans_2 + "[" + c3 + "]的[" + c1 + "]"

        elif rel_2 == "HasProperty":
            rand = random.randint(0, 1)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "有[" + c3 + "]的[" + c1 + "]"
            else:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]是[" + c3 + "]的"

        elif rel_2 == "HasSubevent":
            SurfaceText[sentence] = "在[" + c2 + "]" + trans_1 + "的[" + c1 + "][" + c3 + "]"

        elif rel_2 == "IsA":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]是一種[" + c3 + "]"

        elif rel_2 == "MadeOf":
            if match == 2:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]可以" + trans_2 + "[" + c3 + "]"
            elif match == 3:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]可用[" + c3 + "]" + trans_2

        elif rel_2 == "Have":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "的[" + c1 + "]擁有[" + c3 + "]"

    # 1
    elif rel_1 == "CapableOf":
        if rel_2 == "CapableOf":
            SurfaceText[sentence] = "[" + c1 + "]會[" + c2 + "]和[" + c3 + "]"

        elif rel_2 == "Causes":
            if "] 會帶來 [" in surfacetext_2 or "] 會引發 [" in surfacetext_2:
                SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]後會" + trans_2 + "[" + c3 + "]"
            else:
                SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]後會[" + c3 + "]"

        elif rel_2 == "CausesDesire":
            SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]" + trans_2 + "[" + c3 + "]"

        elif rel_2 == "Desires":
            SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "會[" + c2 + "]的[" + c1 + "]"

        elif rel_2 == "NotDesires":
            if match == 1:
                SurfaceText[sentence] = "[" + c2 + "] [" + c1 + "]時" + trans_2 + "[" + c3 + "]"
            elif match == 2:
                SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "會[" + c2 + "]的[" + c1 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "HasFirstSubevent":
            SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]時要優先[" + c3 + "]"

        elif rel_2 == "HasProperty":
            SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]會[" + c2 + "]"

        elif rel_2 == "HasSubevent":
            if match == 0:
                if "在 [[" in surfacetext_2:
                    SurfaceText[sentence] = "[" + c2 + "]會在[" + c3 + "][" + c1 + "]"
                else:
                    print("No this match\n")
                    match_status = False
            elif match == 1:
                SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]時會[" + c3 + "]"

        elif rel_2 == "IsA":
            SurfaceText[sentence] = "[" + c1 + "]是會[" + c2 + "]的[" + c3 + "]"

        elif rel_2 == "MadeOf":
            rand = random.randint(0, 1)
            if rand == 0:
                SurfaceText[sentence] = "可用[" + c3 + "]" + trans_2 + "的[" + c1 + "]來[" + c2 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "的[" + c1 + "]可以[" + c2 + "]"

        elif rel_2 == "MotivatedByGoal":
            rand = random.randint(0, 1)
            if rand == 0:
                SurfaceText[sentence] = "[" + c2 + "]會為了[" + c3 + "]而[" + c1 + "]"
            else:
                supplement_MotivatedByGoal = Supplement_MotivatedByGoal(c3)
                if supplement_MotivatedByGoal == "來":
                    SurfaceText[sentence] = "[" + c2 + "]會[" + c1 + "]來[" + c3 + "]"
                else:
                    SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]是為了[" + c3 + "]"

        elif rel_2 == "SymbolOf":
            SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]表示[" + c3 + "]"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]時會用到[" + c3 + "]"

        elif rel_2 == "Have":
            rand = random.randint(0, 1)
            if rand == 0:
                SurfaceText[sentence] = "[" + c3 + "]擁有可以[" + c2 + "]的[" + c1 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]可以[" + c2 + "]"

    # 2
    elif rel_1 == "Causes":
        if rel_2 == "Causes":
            if "] 會帶來 [" in surfacetext_1 or "] 會引發 [" in surfacetext_1:
                SurfaceText[sentence] = "因為[" + c2 + "]所" + trans_1 + "的[" + c1 + "]會引發[" + c3 + "]"
            elif "] 後會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "]後[" + c1 + "]會" + trans_2 + "[" + c3 + "]"
            else:
                SurfaceText[sentence] = "因為[" + c2 + "]所以[" + c1 + "]後會[" + c3 + "]"

        elif rel_2 == "CausesDesire":
            if "] 後會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "]後[" + c1 + "]" + trans_2 + "[" + c3 + "]"
            else:
                SurfaceText[sentence] = "因為[" + c2 + "]而[" + c1 + "]" + trans_2 + "[" + c3 + "]"

        elif rel_2 == "Desires":
            if match == 2:
                rand = random.randint(0, 1)
                if rand == 0:
                    SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "會" + trans_1 + "[" + c2 + "]的[" + c1 + "]"
                else:
                    SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "[" + c1 + "]，會" + trans_1 + "[" + c2 + "]"

            elif match == 0:
                rand = random.randint(0, 1)
                if "] 後會 [" in surfacetext_1:
                    if rand == 0:
                        SurfaceText[sentence] = "[" + c3 + "][" + c2 + "]後會想要[" + c1 + "]"
                    else:
                        SurfaceText[sentence] = "[" + c3 + "]想要[" + c2 + "]後[" + c1 + "]"
                else:
                    SurfaceText[sentence] = "[" + c3 + "]因為[" + c2 + "]所以想要[" + c1 + "]"

        elif rel_2 == "NotDesires":
            if match == 0:
                if "] 後會 [" in surfacetext_1:
                    SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "[" + c2 + "]後[" + c1 + "]"
                else:
                    print("No this match\n")
                    match_status = False
            elif match == 2:
                if "] 的時候" in surfacetext_2:
                    if "] 後會 [" not in surfacetext_1:
                        SurfaceText[sentence] = "[" + c3 + "]時" + trans_2 + "[" + c1 + "]會" + trans_1 + "[" + c2 + "]"
                    else:
                        SurfaceText[sentence] = "[" + c3 + "]時" + trans_2 + "[" + c1 + "]後[" + c2 + "]"
                else:
                    SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "會" + trans_1 + "[" + c2 + "]的[" + c1 + "]"

        elif rel_2 == "HasFirstSubevent":
            if "] 後會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c3 + "]時要優先[" + c1 + "]，再[" + c2 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]時要優先[" + c1 + "]，才會" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "HasProperty":
            if "] 後會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]後會[" + c2 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]會" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "HasSubevent":
            if match == 1:
                if "] 後會 [" in surfacetext_1:
                    SurfaceText[sentence] = "[" + c2 + "]後[" + c1 + "]時會[" + c3 + "]"
                else:
                    SurfaceText[sentence] = "因為[" + c2 + "]而[" + c1 + "]時會[" + c3 + "]"
            elif match == 2:
                if "] 的時候會 [" in surfacetext_2:
                    SurfaceText[sentence] = "[" + c3 + "]時[" + c1 + "]會" + trans_1 + "[" + c2 + "]"
                elif "在 [" in surfacetext_2:
                    if "] 後會 [" in surfacetext_1:
                        SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]後會[" + c2 + "]"
                    else:
                        SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]會" + trans_1 + "[" + c2 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "IsA":
            if "] 後會 [" in surfacetext_1:
                print("No this match\n")
                match_status = False
            else:
                SurfaceText[sentence] = "[" + c1 + "]是會" + trans_1 + "[" + c2 + "]的[" + c3 + "]"

        elif rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "的[" + c1 + "]會" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "MotivatedByGoal":
            if "] 後會 [" in surfacetext_1:
                print("No this match\n")
                match_status = False
            else:
                rand = random.randint(0, 1)
                if rand == 0:
                    SurfaceText[sentence] = "為了[" + c3 + "]而[" + c1 + "]會" + trans_1 + "[" + c2 + "]"
                else:
                    SurfaceText[sentence] = "[" + c1 + "]是為了[" + c3 + "]會" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "用[" + c3 + "][" + c1 + "]後會" + "[" + c2 + "]"

    # 3
    elif rel_1 == "CausesDesire":
        if rel_2 == "CausesDesire":
            SurfaceText[sentence] = "[" + c1 + "]會令人想要[" + c2 + "]、[" + c3 + "]"

        elif rel_2 == "NotDesires":
            if "] 會令人想要 [" in surfacetext_1:
                if "] 的時候" in surfacetext_2:
                    SurfaceText[sentence] = "[" + c3 + "]時" + trans_2 + "[" + c1 + "]會令人想要[" + c2 + "]"
                else:
                    SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "[" + c1 + "]會令人想要[" + c2 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "HasProperty":
            # SurfaceText[sentence] = "[" + c1 + "]是[" + c3 + "]的，會令人想要[" + c2 + "]"
            SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "HasSubevent":
            if "] 的時候會 [" in surfacetext_2:
                SurfaceText[sentence] = "[" + c3 + "][" + c1 + "]" + trans_1 + "[" + c2 + "]"
            elif "在 [" in surfacetext_2:
                SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]" + trans_1 + "[" + c2 + "]"

        elif rel_2 == "IsA":
            SurfaceText[sentence] = "[" + c1 + "]是會令人想要[" + c2 + "]的[" + c3 + "]"

        elif rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "的[" + c1 + "]會令人想要[" + c2 + "]"

        elif rel_2 == "MotivatedByGoal":
            if match == 1:
                supplement_MotivatedByGoal = Supplement_MotivatedByGoal(c3)
                SurfaceText[sentence] = "[" + c2 + "]時會想要[" + c1 + "]" + supplement_MotivatedByGoal+ "[" + c3 + "]"
            elif match == 3:
                SurfaceText[sentence] = "[" + c1 + "]是為了[" + c3 + "]，會令人想要[" + c2 + "]"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "用[" + c3 + "][" + c1 + "]時會想要" + "[" + c2 + "]"

        elif rel_2 == "Have":
            rand = random.randint(0, 1)
            if rand == 0:
                SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]會令人想要[" + c2 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]有會令人想要[" + c2 + "]的[" + c1 + "]"

    # 4
    elif rel_1 == "Desires":
        if rel_2 == "Desires":
            SurfaceText[sentence] = "[" + c1 + "]" + trans_1 + "[" + c2 + "]和[" + c3 + "]"

        elif rel_2 == "NotDesires":
            SurfaceText[sentence] = "[" + c1 + "]" + trans_1 + "[" + c2 + "]但" + trans_2 + "[" + c3 + "]"

        elif rel_2 == "HasFirstSubevent":
            if "] 想要 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "]想要[" + c1 + "]時，要優先[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "HasProperty":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c3 + "]的[" + c1 + "]"

        elif rel_2 == "HasSubevent":
            if "] 想要 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "]想要[" + c1 + "]時會[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "IsA":
            if "] 喜歡 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c1 + "]是喜歡[" + c2 + "]的[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c3 + "]" + trans_2 + "的[" + c1 + "]"

        elif rel_2 == "MotivatedByGoal":
            rand = random.randint(0, 1)
            if rand == 0:
                supplement_MotivatedByGoal = Supplement_MotivatedByGoal(c3)
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c1 + "]" + supplement_MotivatedByGoal + "[" + c3 + "]"
            else:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "為了[" + c3 + "]而[" + c1 + "]"

        elif rel_2 == "Have":
            SurfaceText[sentence] = "[" + c1 + "]擁有[" + c3 + "]，" + trans_1 + "[" + c2 + "]"

    # 5
    elif rel_1 == "NotDesires":
        if rel_2 == "NotDesires":
            if match == 0:
                SurfaceText[sentence] = "[" + c2 + "]和[" + c3 + "]" + trans_1 + "[" + c1 + "]"
            elif match == 3:
                SurfaceText[sentence] = "[" + c1 + "]" + trans_1 + "[" + c2 + "]和[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "HasProperty":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c3 + "]的[" + c1 + "]"

        elif rel_2 == "HasSubevent":
            if match == 1:
                SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c1 + "]時[" + c3 + "]"
            elif match == 2:
                if "] 的時候" in surfacetext_1:
                    if "在 [" in surfacetext_2:
                        SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]時" + trans_1 + "[" + c2 + "]"
                    else:
                        SurfaceText[sentence] = "[" + c3 + "]時[" + c1 + "]" + trans_1 + "[" + c2 + "]"
                else:
                    print("No this match\n")
                    match_status = False

        elif rel_2 == "IsA":
            SurfaceText[sentence] = "[" + c1 + "]是" + trans_1 + "[" + c2 + "]的[" + c3 + "]"

        elif rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c2 + "]" + trans_1 + "[" + c3 + "]" + trans_2 + "的[" + c1 + "]"

        elif rel_2 == "Have":
            SurfaceText[sentence] = "擁有[" + c3 + "]的[" + c1 + "]" + trans_1 + "[" + c2 + "]"

    # 6
    elif rel_1 == "HasFirstSubevent":
        if rel_2 == "HasSubevent":
            if "] 的時候會 [" in surfacetext_2:
                if match == 2:
                    SurfaceText[sentence] = "[" + c3 + "]時[" + c1 + "]要優先[" + c2 + "]"
                elif match == 3:
                    SurfaceText[sentence] = "[" + c1 + "]時要優先[" + c2 + "]再[" + c3 + "]"
            else:
                SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]時要優先[" + c2 + "]"

        elif rel_2 == "MotivatedByGoal":
            if match == 0:
                SurfaceText[sentence] = "[" + c2 + "]時要優先[" + c3 + "][" + c1 + "]"
            elif match == 1:
                supplement_MotivatedByGoal = Supplement_MotivatedByGoal(c3)
                SurfaceText[sentence] = "[" + c2 + "]時要優先[" + c1 + "]" + supplement_MotivatedByGoal + "[" + c3 + "]"
            elif match == 2:
                SurfaceText[sentence] = "[" + c3 + "][" + c1 + "]時要優先[" + c2 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "使用[" + c3 + "][" + c1 + "]時，要優先[" + c2 + "]"

    # 7
    elif rel_1 == "HasProperty":
        if rel_2 == "HasProperty":
            SurfaceText[sentence] = "[" + c1 + "]是[" + c2 + "]、[" + c3 + "]的"

        elif rel_2 == "HasSubevent":
            if "在 [" in surfacetext_2:
                SurfaceText[sentence] = "在[" + c2 + "]的[" + c1 + "][" + c3 + "]"
            else:
                SurfaceText[sentence] = "[" + c3 + "]時會[" + c2 + "]的[" + c1 + "]"

        elif rel_2 == "IsA":
            SurfaceText[sentence] = "[" + c1 + "]是[" + c2 + "]的[" + c3 + "]"

        elif rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c3 + "]" + trans_2 + "的[" + c1 + "]是[" + c2 + "]的"

        elif rel_2 == "MotivatedByGoal":
            if match == 2:
                SurfaceText[sentence] = "[" + c3 + "]是為了[" + c2 + "]的[" + c1 + "]"
            elif match == 3:
                SurfaceText[sentence] = "[" + c2 + "]的[" + c1 + "]是為了[" + c3 + "]"

        elif rel_2 == "PartOf":
            SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]是[" + c2 + "]的"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "[" + c3 + "]時會用到[" + c2 + "]的[" + c1 + "]"

        elif rel_2 == "Have":
            if match == 2:
                SurfaceText[sentence] = "[" + c3 + "]擁有[" + c2 + "]的[" + c1 + "]"
            elif match == 3:
                SurfaceText[sentence] = "擁有[" + c3 + "]的[" + c1 + "]是[" + c2 + "]的"

    # 8
    elif rel_1 == "HasSubevent":
        if rel_2 == "HasSubevent":
            if "在 [" in surfacetext_2:
                if match == 2:
                    SurfaceText[sentence] = "在[" + c3 + "][" + c1 + "]時會[" + c2 + "]"
                elif match == 3:
                    SurfaceText[sentence] = "在[" + c1 + "]時會[" + c2 + "]和[" + c3 + "]"
            else:
                if match == 2:
                    SurfaceText[sentence] = "[" + c3 + "]時會[" + c2 + "][" + c1 + "]"
                elif match == 3:
                    SurfaceText[sentence] = "[" + c1 + "]時會[" + c2 + "]和[" + c3 + "]"

        elif rel_2 == "MotivatedByGoal":
            if "] 的時候會 [" in surfacetext_1:
                supplement_MotivatedByGoal = Supplement_MotivatedByGoal(c3)
                if supplement_MotivatedByGoal == "來":
                    SurfaceText[sentence] = "[" + c2 + "]時會[" + c1 + "]來[" + c3 + "]"
                else:
                    SurfaceText[sentence] = "[" + c2 + "]時[" + c1 + "]是為了[" + c3 + "]"
            elif "在 [" in surfacetext_1:
                SurfaceText[sentence] = "在[" + c2 + "][" + c1 + "]是為了[" + c3 + "]"

        elif rel_2 == "SymbolOf":
            if "] 的時候會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "]時[" + c1 + "]代表[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "MayUse":
            if "] 的時候會 [" in surfacetext_1:
                SurfaceText[sentence] = "[" + c2 + "][" + c1 + "]時會用到[" + c3 + "]"
            elif "在 [" in surfacetext_1:
                SurfaceText[sentence] = "在[" + c2 + "][" + c1 + "]時會用到[" + c3 + "]"

    # 9
    elif rel_1 == "IsA":
        if rel_2 == "MadeOf":
            if match == 3:
                SurfaceText[sentence] = "[" + c1 + "]是[" + c3 + "]" + trans_2 + "的[" + c2 + "]"
            elif match == 2:
                SurfaceText[sentence] = "[" + c1 + "]是一種可以" + trans_2 + "[" + c3 + "]的[" + c2 + "]"

        elif rel_2 == "PartOf":
            if match == 1:
                SurfaceText[sentence] = "[" + c2 + "]是[" + c3 + "]的一小部分"
            elif match == 3:
                SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]是[" + c2 + "]"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "[" + c3 + "]時會用到[" + c2 + "]"

        elif rel_2 == "Have":
            SurfaceText[sentence] = "[" + c1 + "]是有[" + c3 + "]的[" + c2 + "]"

    # 10
    elif rel_1 == "MadeOf":
        if rel_2 == "MadeOf":
            SurfaceText[sentence] = "[" + c1 + "]可由[" + c2 + "]和[" + c3 + "]" + trans_2

        elif rel_2 == "MotivatedByGoal":
            SurfaceText[sentence] = "[" + c3 + "]是為了[" + c2 + "]" + trans_1 + "的[" + c1 + "]"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "[" + c3 + "]時會用到[" + c2 + "]" + trans_1 + "的[" + c1 + "]"

        elif rel_2 == "Have":
            SurfaceText[sentence] = "[" + c3 + "]擁有[" + c2 + "]" + trans_1 + "的[" + c1 + "]"

    # 11
    elif rel_1 == "MotivatedByGoal":
        if rel_2 == "SymbolOf":
            SurfaceText[sentence] = "為了[" + c2 + "]而[" + c1 + "]代表[" + c3 + "]"

    # 12
    elif rel_1 == "PartOf":
        if rel_2 == "SymbolOf":
            SurfaceText[sentence] = "[" + c3 + "]的[" + c1 + "]是[" + c2 + "]的一小部分"

        elif rel_2 == "MayUse":
            SurfaceText[sentence] = "[" + c3 + "]時會用到[" + c2 + "]的[" + c1 + "]"

    # 13
    elif rel_1 == "SymbolOf":
        if rel_2 == "Have":
            SurfaceText[sentence] = "[" + c3 + "]擁有代表[" + c2 + "]的[" + c1 + "]"

    # 14
    elif rel_1 == "MayUse":
        if rel_2 == "MayUse":
            if match == 0:
                SurfaceText[sentence] = "[" + c2 + "]和[" + c3 + "]時會用到[" + c1 + "]"
            elif match == 3:
                SurfaceText[sentence] = "[" + c1 + "]時會用到[" + c2 + "]和[" + c3 + "]"
            else:
                print("No this match\n")
                match_status = False

        elif rel_2 == "Have":
            SurfaceText[sentence] = "[" + c3 + "]有[" + c1 + "]，[" + c2 + "]時會用到"

    # 15
    elif rel_1 == "Have":
        if rel_2 == "Have":
            SurfaceText[sentence] = "[" + c1 + "]擁有[" + c2 + "]和[" + c3 + "]"

    return match_status


""" Node class """
class Node():
    def __init__(self, *arg):
        self.parent_ = None
        self.children_ = list()
        self.data_ = ""
        self.relation_ = ""
        self.mode_ = -1
        self.score_ = 0
        self.visit_count_ = 0
        self.search_position_ = ""
        self.terminal_ = False
        self.first_selection_ = False
        # print("arg content:", arg)
        if not arg:
            self.data_ = ""
            # print("No arg")
        elif len(arg) == 1:
            arg = arg[0]
            # print("arg class:", arg.__class__.__name__)
            if arg.__class__.__name__ == "str":
                self.data_ = arg
                # print("str arg")
            elif arg.__class__.__name__ == "Node":
                node = arg
                self.data_ = node.data_
                self.relation_ = node.relation_
                self.mode_ = node.mode_
                self.score_ = node.score_
                self.visit_count_ = node.visit_count_
                self.search_position_ = node.search_position_
                self.terminal_ = node.terminal_
                # print("Node arg")

    def data(self):
        return self.data_
    def set_data(self, data):
        self.data_ = data
    def relation(self):
        return self.relation_
    def set_relation(self, rel):
        self.relation_ = rel
    def mode(self):
        return self.mode_
    def set_mode(self, node_mode):
        self.mode_ = node_mode
    def score(self):
        return self.score_
    def add_score(self, score):
        self.score_ = self.score_ + score
    # Select children with topN max score
    def selectChildren(self, target_sentiment):
        total_visit_count = mean_score = 0
        selected_node = Node()
        nodes_dict = dict()

        file_w.write("parent:\n" + self.data_ + "(" + str(self.visit_count_) + "):" + str(round(self.score_/self.visit_count_, 3)) + '\n')
        file_w.write("select node:\n")
        for child in self.children_:
            if child.visit_count_ == 0:
                continue
            mean_score = round(child.score_ / child.visit_count_, 3)
            nodes_dict[child] = mean_score
            file_w.write(child.data_ + "(" + str(child.visit_count_) + "):" + str(mean_score) + '\n')
            total_visit_count += child.visit_count_
        file_w.write("total_visit_count:" + str(total_visit_count) + '\n')

        if not nodes_dict:
            return selected_node

        nodes_dict = dict(sorted(nodes_dict.items(), key=lambda x:x[1], reverse=True))  # sorted by node's score in descending order

        # if the concept has the same sentiment as target_sentiment, move up the ranks
        nodes_list = list(nodes_dict.keys())
        nodes_list, same_sentiment_count = sameSentiment(nodes_list, target_sentiment)

        # select top N percent max score children randomly
        children_size = len(self.children_)
        if children_size >= 1000:
            proportion = 0.05
        elif children_size >= 500:
            proportion = 0.1
        elif children_size >= 250:
            proportion = 0.15
        else:
            proportion = 0.2
        topN = int(round(children_size*proportion, 0))
        if topN == 0:
            topN = 1
        if topN > same_sentiment_count and same_sentiment_count >= 5:  # avoid selecting opposite sentiment concept
            topN = same_sentiment_count

        selected_node = random.choice(nodes_list[:topN])

        file_w.write("select:\n" + selected_node.data_ + "(" + str(selected_node.visit_count_) + "):" + str(round(selected_node.score_/selected_node.visit_count_, 3)) + '\n')

        # test
        file_w.write("\nselected node's children:\n")
        for child in selected_node.children_:
            mean_score = 0
            if child.visit_count_ != 0:
                mean_score = round(child.score_ / child.visit_count_, 3)
            file_w.write(child.data_ + "(" + str(child.visit_count_) + "):" + str(mean_score) + '\n')
        # test

        return selected_node

    def visit_count(self):
        return self.visit_count_
    def incrementVisitCount(self):
        self.visit_count_ += 1
    def isVisited(self):
        return True if self.visit_count_ > 0 else False
    def search_position(self):
        return self.search_position_
    def set_search_position(self, search_position):
        self.search_position_ = search_position
    def isTerminal(self):
        return self.terminal_
    def set_terminal(self, terminal):
        self.terminal_ = terminal
    def parent(self):
        return self.parent_
    def getRoot(self):
        while self.parent_:
            self = self.parent_
        return self
    def getRootSentence(self, depth):
        while self.parent_:
            self = self.parent_
            depth -= 1
        sentence = currentSentence(depth)
        return sentence
    def children(self):
        return self.children_
    def addChild(self, child):
        if not child:
            sys.stderr.write("can't insert null value!!\n")
        else:
            self.children_.append(child)
            child.parent_ = self
    def siblingSize(self):
        return len(self.parent_.children_)-1
    # delete the link between parent and child(not deleting 'that' node)
    def delete(self):
        self.parent_.children_.remove(self)
        self.resetNode()
    def getRandomPossibleMove(self):
        return self.children_[random.randint(0, len(self.children_)-1)]
    def isLeaf(self):
        return True if not self.children_ else False
    def resetNode(self):
        self.parent_ = None
        for child in self.children_:
            child.parent_ = None
        self.children_.clear()
        self.data_ = ""
        self.relation_ = ""
        self.mode_ = -1
        self.score_ = 0
        self.visit_count_ = 0
        self.search_position_ = ""
        self.terminal_ = False
    # Preorder
    def display(self, node, appender):
        if not node:
            return
        print(appender, node.data_, sep='')
        for child in node.children_:
            self.display(child, appender + "  ")
    def setFirstSelection(self, flag):
        self.first_selection_ = flag
    def firstSelection(self):
        return self.first_selection_



""" Neural Network """
def readVocabList_freq(freq):
    vocab_list = list()
    vocab_list_path = CURRENT_PATH / r"data\ptt_wiki_frequency_noHMM.txt"
    with open(vocab_list_path, 'r', encoding="UTF-8") as f:
        for line in f:
            line = line.replace(' ', '')
            line = line.split(':')

            #詞頻大於input freq
            if int(line[1]) > freq:
                vocab_list.append(line[0])
    vocab_list.sort()
    return vocab_list


def loadPretrainedEmbedding(vocab_list_30):
    global embedding_model, WE_DIMENSIONS
    # embedding_model_path = "D:\\Andy\\研究所\\research\\model\\word_embedding\\ptt_wiki\\prediction_based\\best_model\\word2vec_500d_CBOW_alpha0025_sample000001_neg2_iter5.model"
    embedding_model_path = CURRENT_PATH / r"model\SVD_700d_ws3_p0.5_SPPMI_k10_skip6_321.vec"
    embedding_model = KeyedVectors.load_word2vec_format(embedding_model_path, binary=True)
    WE_DIMENSIONS = embedding_model.vector_size

    for i, word in enumerate(vocab_list_30):
        word = word.strip()
        word2idx[word] = i + 1
        idx2word[i+1] = word


def predictedData(pred_paragraph):
    global total_paragraph_num
    global invalid_sent_num

    pred_x = np.zeros((MAX_SENTENCE_NUM, MAX_WORD_NUM), dtype="int")
    total_paragraph_num += 1
    seg_num = 0

    file_w.write("----------\n")
    for s, sentence in enumerate(pred_paragraph):
        sentence = sentence.replace('，', '').split(' ')
        for word in sentence:
            if word in word2idx:
                seg_num += 1
                if seg_num > MAX_WORD_NUM-1:  # 避免一行裡的seg_num大於MAX_WORD_NUM而無法處理的錯誤
                    invalid_sent_list.append(sentence)
                    invalid_sent_num += 1
                    break
                file_w.write(word+' ')
                pred_x[s, seg_num-1] = word2idx[word]
        seg_num = 0
        file_w.write('\n')
    file_w.write("----------\n")
    return pred_x


def predictScore(pred_paragraph):
    pred_x = np.zeros((1, MAX_SENTENCE_NUM, MAX_WORD_NUM), dtype="int")
    pred_x[0] = predictedData(pred_paragraph)
    pred_y = NN_model.predict(pred_x,
                              batch_size = 1,
                              verbose = 0,
                              steps = None)
    score = pred_y[0,0].round(decimals = 3)
    return score


if __name__ == "__main__":
    main()
