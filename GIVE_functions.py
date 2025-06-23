import openai
from tqdm import tqdm
import os
from collections import Counter
import pickle
import torch
import json
from prompts import *
from os import walk
import xmltodict
from sentence_transformers import SentenceTransformer, util
import pyarrow.parquet as pa
import pandas
import argparse
import warnings
warnings.filterwarnings('ignore')
def load_processbank():
    filenames = next(walk("data/QA/processbank/qa"), (None, None, []))[2]
    path = "data/QA/processbank/qa/"
    questions = []
    choices = []
    keys = []
    for file in filenames:
        file_path = path + file
        with open(file_path, 'r', encoding='utf-8') as file:
            my_xml = file.read()

            try:
                my_dict = xmltodict.parse(my_xml)["annotation"]["questions"]["question"][0]
            except Exception:
                my_dict = xmltodict.parse(my_xml)["annotation"]["questions"]["question"]
            #exit()
            questions.append(my_dict['q'])
            choice = "A."
            choice += my_dict['a0']
            choice += ". B."
            choice += my_dict['a1']
            choice += "."
            choices.append(choice)
            key = my_dict['correct']
            if(key == "0"): key = "A"
            elif(key == "1"): key = "B"
            else:
                print("Error in loading keys")
                exit()
            keys.append(key)
    return questions, choices, keys

def load_bioasq():
    # Opening JSON file
    f1 = open('data/QA/BioASQ/Task2BGoldenEnriched/2B1_golden.json')
    f2 = open('data/QA/BioASQ/Task2BGoldenEnriched/2B2_golden.json')
    f3 = open('data/QA/BioASQ/Task2BGoldenEnriched/2B3_golden.json')
    f4 = open('data/QA/BioASQ/Task2BGoldenEnriched/2B4_golden.json')
    f5 = open('data/QA/BioASQ/Task2BGoldenEnriched/2B5_golden.json')
    f6 = open('data/QA/BioASQ/Task3BGoldenEnriched/3B1_golden.json')
    f7 = open('data/QA/BioASQ/Task3BGoldenEnriched/3B2_golden.json')
    f8 = open('data/QA/BioASQ/Task3BGoldenEnriched/3B3_golden.json')
    f9 = open('data/QA/BioASQ/Task3BGoldenEnriched/3B4_golden.json')
    f10 = open('data/QA/BioASQ/Task3BGoldenEnriched/3B5_golden.json')
    f11 = open('data/QA/BioASQ/Task4BGoldenEnriched/4B1_golden.json')
    f12 = open('data/QA/BioASQ/Task4BGoldenEnriched/4B2_golden.json')
    f13 = open('data/QA/BioASQ/Task4BGoldenEnriched/4B3_golden.json')
    f14 = open('data/QA/BioASQ/Task4BGoldenEnriched/4B4_golden.json')
    f15 = open('data/QA/BioASQ/Task4BGoldenEnriched/4B5_golden.json')

    # returns JSON object as
    # a dictionary
    datas = json.load(f1)['questions']
    datas.extend(json.load(f2)['questions'])
    datas.extend(json.load(f3)['questions'])
    datas.extend(json.load(f4)['questions'])
    datas.extend(json.load(f5)['questions'])
    datas.extend(json.load(f6)['questions'])
    datas.extend(json.load(f7)['questions'])
    datas.extend(json.load(f8)['questions'])
    datas.extend(json.load(f9)['questions'])
    datas.extend(json.load(f10)['questions'])
    datas.extend(json.load(f11)['questions'])
    datas.extend(json.load(f12)['questions'])
    datas.extend(json.load(f13)['questions'])
    datas.extend(json.load(f14)['questions'])
    datas.extend(json.load(f15)['questions'])
    questions = []
    answers = []
    for data in datas:
        if("exact_answer" in list(data.keys())):
            if(data["exact_answer"] == "yes" or data["exact_answer"] == "no"):
                questions.append(data["body"])
                answers.append(data["exact_answer"])
    return questions, answers
def load_pubmedQA():
    file_path = "data/QA/pubmedqa/ori_pqal.json"
    key_path = "data/QA/pubmedqa/test_ground_truth.json"
    with open(file_path) as json_file:
        question_data = json.load(json_file)
    with open(key_path) as json_file:
        key_data = json.load(json_file)
    return question_data, key_data

def load_commonsenseqa():
    with open('data/QA/commonsenseqa/dev_rand_split.jsonl', 'r') as json_file:
        json_list = list(json_file)
    concepts = []
    choices = []
    questions = []
    ids = []
    keys = {}
    for json_str in json_list:
        instance = json.loads(json_str)
        concept = instance['question']['question_concept']
        choice = instance['question']['choices']
        question = instance['question']['stem']
        choice_str = ""
        for choice_instance in choice:
            choice_str += choice_instance['label']
            choice_str += ":"
            choice_str += choice_instance['text']
            choice_str += " "
        concepts.append(concept)
        choices.append(choice_str)
        questions.append(question)
        ids.append(instance['id'])
        keys[instance['id']] = instance["answerKey"]
    return keys, ids, concepts, choices, questions

def build_group(source_entity,k, sentence_transformer_id):
    group = [source_entity]
    landing_candidates = semantic_top_k([source_entity], k, sentence_transformer_id)
    group.extend(landing_candidates)
    return group

def semantic_top_k(Qs, k, sentence_transformer_id):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('data/KG/umls/txt2entity.pkl', 'rb') as f1:
        txt2concepts = pickle.load(f1)
    encoder_model = SentenceTransformer(sentence_transformer_id)
    query_embedding = encoder_model.encode(Qs, convert_to_tensor=True).to(device)
    concept_list = list(txt2concepts.keys())
    concept_embeddings = encoder_model.encode(concept_list, convert_to_tensor=True).to(device)
    cosine_scores = util.cos_sim(query_embedding, concept_embeddings)
    cosine_scores = torch.sum(cosine_scores,0)
    cosine_scores = torch.squeeze(cosine_scores)
    sorted_idx = torch.argsort(cosine_scores, descending=True)
    sorted_idx = sorted_idx.tolist()[:k]
    results = []
    for idx in sorted_idx:
        results.append(concept_list[idx].lower())
    return results

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0.2):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    answer =  response.choices[0].message.content
    return answer

def rewrite_question(Q, model, temperature):
    messages = [
        {'role': 'system',
         'content': rewrite_question_system_msg},
        {'role': 'user',
         'content': rewrite_question_prompts + Q
         },
        {
            'role': "assistant",
            'content': "Re-written question: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    return response



def get_top_entities(Q, model, temperature):
    messages = [
        {'role': 'system',
         'content': get_entities_system_msg},
        {'role': 'user',
         'content': get_entities_prompts + Q
         },
        {
            'role': "assistant",
            'content': "Entities: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    entity_list = response.split(", ")
    return entity_list

def get_top_entities_processbank(Q, choice, model, temperature):
    messages = [
        {'role': 'system',
         'content': get_entities_processbank_system_msg},
        {'role': 'user',
         'content': get_entities_processbank_prompts + "Question: " + Q + "\n" + "Choices: " + choice
         },
        {
            'role': "assistant",
            'content': "Entities: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    entity_list = response.split(", ")
    return entity_list

def get_top_relations(Q, model, temperature):
    messages = [
        {'role': 'system',
         'content': get_relations_system_msg},
        {'role': 'user',
         'content': get_relations_prompts + Q
         },
        {
            'role': "assistant",
            'content': "Relationship or attributes:: {}"

        }

    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.lower()
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    relationship_list = response.split(", ")
    return relationship_list

def get_top_relations_processbank(Q, choice, model, temperature):
    messages = [
        {'role': 'system',
         'content': get_relations_processbank_system_msg},
        {'role': 'user',
         'content': get_relations_processbank_prompts + "Question: " + Q + "\n" + "Choices: " + choice
         },
        {
            'role': "assistant",
            'content': "Relationship or attributes:: {}"

        }

    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.lower()
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    relationship_list = response.split(", ")
    return relationship_list





def add_relation_internal_knowledge(entities_txt, model, temperature):
    messages = [
        {'role': 'system',
         'content': internal_knowledge_system_msg},

        {'role': 'user',
         'content': internal_knowledge_prompts + entities_txt},
        {
            'role': "assistant",
            'content': "Relationship: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    return response.lower()


def get_relation_between_groups_KG(group1, group2):
    with open('data/KG/umls/txt2entity.pkl', 'rb') as f1:
        txt2entity = pickle.load(f1)
    with open('data/KG/umls/entity2txt.pkl', 'rb') as f1:
        entity2txt = pickle.load(f1)
    with open('data/KG/umls/relation2txt.pkl', 'rb') as f1:
        relation2txt = pickle.load(f1)
    G = pickle.load(open('data/KG/umls/umls_nx.pickle', 'rb'))
    source_entities = []
    target_entities = []
    for source_entity_candidate in group1:
        try:
            source_entities.append(txt2entity[source_entity_candidate])
        except Exception:
            continue

    for target_entity_candidate in group2:
        try:
            target_entities.append(txt2entity[target_entity_candidate])
        except Exception:
            continue
    #source_entities = [txt2entity[txt] for txt in group1[1:]]
    #target_entities = [txt2entity[txt] for txt in group2[1:]]
    edges = [(entity2txt[u],relation2txt[d['relationship']],entity2txt[v]) for u,v,d in G.edges.data() if u in source_entities and v in target_entities]
    relations = [r for (e1,r,e2) in edges]
    return edges, relations


def build_connections(query, group1, group2, query_relations, model, consider_intermediate, sentence_transformer_id,
                      entity_per_group, temperature):
    """return all knowledge triplets from group1 to group2"""
    yes_candidates = []
    no_candidates = []
    maybe_candidates = []
    self_knowledge = []

    candidate_relations = []
    KG_knowledge, KG_relations = get_relation_between_groups_KG(group1, group2)
    candidate_relations.extend(KG_relations)
    candidate_relations.extend(query_relations)
    candidate_relations = list(set(candidate_relations))

    for i in range(len(group1)):
        for j in range(len(group2)):
            for relation in candidate_relations:
                self_knowledge_flag = query_if_relation_exists(group1[i], relation, group2[j], model, temperature)
                if (self_knowledge_flag == "Y"):
                    yes_candidates.append((group1[i], relation, group2[j]))
                elif (self_knowledge_flag == "M"):
                    maybe_candidates.append((group1[i], "maybe " + relation, group2[j]))
                else:
                    no_candidates.append((group1[i], "not " + relation, group2[j]))

            txt = ""
            txt += group1[i]
            txt += ", "
            txt += group2[j]
            candidate_relation = add_relation_internal_knowledge(txt, model, temperature)
            if ('unrelated' in candidate_relation or 'does not contain' in candidate_relation):
                continue
            else:
                self_knowledge.append((group1[i], candidate_relation, group2[j]))

    if (consider_intermediate and len(yes_candidates) == 0):
        intermediate_node = get_intermediate_node(group1, group2, query, model, temperature)
        if (intermediate_node != ""):
            intermediate_group = build_group(intermediate_node, entity_per_group, sentence_transformer_id)
            intermediate_group = list(set(intermediate_group))
            yes_candidates1, maybe_candidates1, no_candidates1, self_knowledge1, KG_knowledge1 = build_connections(
                query, group1, intermediate_group, query_relations, model, False, sentence_transformer_id, entity_per_group, temperature)
            yes_candidates2, maybe_candidates2, no_candidates2, self_knowledge2, KG_knowledge2 = build_connections(
                query, intermediate_group, group2, query_relations, model, False, sentence_transformer_id, entity_per_group, temperature)
            yes_candidates.extend(yes_candidates1)
            yes_candidates.extend(yes_candidates2)
            maybe_candidates.extend(maybe_candidates1)
            maybe_candidates.extend(maybe_candidates2)
            no_candidates.extend(no_candidates1)
            no_candidates.extend(no_candidates2)
            self_knowledge.extend(self_knowledge1)
            self_knowledge.extend(self_knowledge2)

    return yes_candidates, maybe_candidates, no_candidates, self_knowledge, KG_knowledge


def query_if_relation_exists(entity1, relation, entity2, model, temperature):
    query = "(" + entity1 + ", " + relation + ", " + entity2 + ")"
    messages = [
        {'role': 'system',
         'content': prune_candidate_knowledge_system_msg},

        {'role': 'user',
         'content': prune_candidate_knowledge_prompts + query
         },
        {
            'role': "assistant",
            'content': "Correctness: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature).lower()
    response.replace(".", "")
    response = response.lower()
    if(len(response) == 0):return "M"
    if (response[0] == 'y'):
        return "Y"
    elif(response[0] == 'n'):
        return "N"
    return "M"

def get_intermediate_node(group1, group2, Q, model, temperature):
    G = pickle.load(open('data/KG/umls/umls_nx.pickle', 'rb'))
    with open('data/KG/umls/entity2txt.pkl', 'rb') as f1:
        entity2txt = pickle.load(f1)
    with open('data/KG/umls/relation2txt.pkl', 'rb') as f2:
        relation2txt = pickle.load(f2)
    edges_out_group1 = [(u, d['relationship'], v) for u,v,d in G.edges.data() if entity2txt[u] in group1]
    edges_in_group2 = [(u, d['relationship'], v) for u,v,d in G.edges.data() if entity2txt[v] in group2]
    ziped_edges = list(zip(edges_out_group1, edges_in_group2))
    two_hops = [(a[0],a[1],b[0],b[1],b[2]) for a,b in ziped_edges if a[2] == b[0]]
    two_hops = [(entity2txt[a],relation2txt[b],entity2txt[c], relation2txt[d], entity2txt[e]) for a,b,c,d,e in two_hops]
    if(len(two_hops)== 0):
        return ""
    knowledge_txt = ""
    for path in two_hops:
        knowledge_txt += "("
        knowledge_txt += path[0]
        knowledge_txt += ", "
        knowledge_txt += path[1]
        knowledge_txt += ", "
        knowledge_txt += path[2]
        knowledge_txt += ", "
        knowledge_txt += path[3]
        knowledge_txt += ", "
        knowledge_txt += path[4]
        knowledge_txt += ")"
        knowledge_txt += ", "
    knowledge_txt = knowledge_txt[:-2]
    if(len(knowledge_txt) < 2): return ""
    messages = [
        {'role': 'system',
         'content': intermediate_node_system_msg},

        {'role': 'user',
         'content': intermediate_node_system_prompts+"Question:"+Q+"\n"+"Two-hop facts:"+knowledge_txt},
        {
            'role': "assistant",
            'content': "Most important two-hop facts: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    index = response.find("(")
    if(index == -1): return ""
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    ans = response.split(",")
    if(len(ans) < 3): return ""
    ans = ans[2]
    if(ans[-1] == ")"): ans = ans[:-1]
    if(ans[0] == " "): ans = ans[1:]
    return ans


def verbolize(knowledge_set):
    knowledge_txt = ""
    for triplet in knowledge_set:
        knowledge_txt += "("
        knowledge_txt += triplet[0]
        knowledge_txt += ", "
        knowledge_txt += triplet[1]
        knowledge_txt += ", "
        knowledge_txt += triplet[2]
        knowledge_txt += ")"
        knowledge_txt += ", "
    knowledge_txt = knowledge_txt[:-2]
    return knowledge_txt

def generate_answer_a_pubmedqa(Q, triplets, temperature, model):
    examples = examples_5shot_pubmedqa
    prompt = examples + "\n" + """

                       Q:""" + Q + """. Return only yes, no or maybe.
                       Knowledge triplets: """ + triplets

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes, no or maybe, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge. The return must be one of yes, no or maybe, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.lower()
    return response


def generate_answer_ac_pubmedqa(Q, open_and_yes_txt, no_txt, answer_first, temperature, model):
    examples = examples_5shot_pubmedqa
    prompt = examples+ "\n" + """

                       Q:""" + Q + """. Return only yes, no or maybe.
                       Knowledge triplets: """ + open_and_yes_txt +  "\n" + "A:  "+ answer_first + "\nAdditional knowledge triplets: " + no_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes, no or maybe, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge. The return must be one of yes, no or maybe, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index+1:index2]
    response = response.lower()
    return response

def generate_answer_ace_pubmedqa(Q, open_and_yes_txt, no_txt, KG_txt, answer_first, answer_second, temperature, model):
    examples = examples_5shot_pubmedqa
    prompt = examples + "\n" + """

                       Q:""" + Q + """. Return only yes, no or maybe.
                       Knowledge triplets: """ + open_and_yes_txt +  "\n" + "A:  "+ answer_first + "\nAdditional knowledge triplets: " + no_txt + "\n" + "A: "+answer_second + "\n" + "Additional knowledge triplets retrieved from expert knowledge base: " + KG_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes, no or maybe, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge, and the knowledge triplets from an expert knowledge base. The return must be one of yes, no or maybe, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index+1:index2]
    response = response.lower()
    return response

def generate_answer_a_bioasq(Q, triplets, temperature, model):
    prompt = examples_5shot_bioasq + "\n" + """

                       Q:""" + Q + """. Only return yes or no.
                       Knowledge triplets: """ + triplets

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes or no, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge. The return must only be one of yes or no, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    return response.lower()


def generate_answer_ac_bioasq(Q, open_and_yes_txt, no_txt, answer_first, temperature, model):
    prompt = examples_5shot_bioasq + "\n" + """

                       Q:""" + Q + """. Only return yes or no.
                       Knowledge triplets: """ + open_and_yes_txt + "\n" + "A:  " + answer_first + "\nAdditional knowledge triplets: " + no_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes or no, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge. The return must only be one of yes or no, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    return response.lower()


def generate_answer_ace_bioasq(Q, open_and_yes_txt, no_txt, KG_txt, answer_first, answer_second, temperature,
                                     model):
    prompt = examples_5shot_bioasq + "\n" + """

                       Q:""" + Q + """. Only return yes or no.
                       Knowledge triplets: """ + open_and_yes_txt + "\n" + "A:  " + answer_first + "\nAdditional knowledge triplets: " + no_txt + "\n" + "A: " + answer_second + "\n" + "Additional knowledge triplets retrieved from expert knowledge base: " + KG_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given question about medical knowledge with yes or no, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge, and the knowledge triplets from an expert knowledge base. The return must only be one of yes or no, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    return response.lower()

def generate_answer_a_processbank(Q, choice, triplets, temperature, model):
    prompt = examples_10shot_processbank + "\n" + """"
    
            Q:""" + Q + """ Only return A or B.
            Choices:""" + choice + """
            Knowledge triplets: """ + triplets

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given multiple choice question about medical knowledge with A or B, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge, and the knowledge triplets from an expert knowledge base. The return must be one of A or B, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response



def generate_answer_ac_processbank(Q, choice, open_and_yes_txt, no_txt, answer_first, temperature, model):
    prompt = examples_10shot_processbank + "\n" + """
    
            Q:""" + Q + """ Only return A or B."""+"""
            Choices:""" + choice + """
            Knowledge triplets: """ + open_and_yes_txt + "\n" + "A:  "+ answer_first + "\nAdditional knowledge triplets: " + no_txt
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given multiple choice question about medical knowledge with A or B, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge, and the knowledge triplets from an expert knowledge base. The return must be one of A or B, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response



def generate_answer_ace_processbank(Q, choice, open_and_yes_txt, no_txt, KG_txt, answer_first, answer_second, temperature,
                                     model):
    prompt = examples_10shot_processbank + "\n" + """
    
            Q:""" + Q + """ Only return A or B."""+ """
            Choices:""" + choice + """
            Knowledge triplets: """ + open_and_yes_txt +  "\n" + "A:  "+ answer_first + "\nAdditional knowledge triplets: " + no_txt + "\n" + "A: "+answer_second + "\n" + "Additional knowledge triplets retrieved from expert knowledge base: " + KG_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given multiple choice question about medical knowledge with A or B, based on the retrieved knowledge triplets (entity, relation, entity) from your own knowledge, and the knowledge triplets from an expert knowledge base. The return must be one of A or B, no explanation needed."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index+1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response



























def GIVE_a_answer(dataset, Q, affirmative_knowledge, model, temperature):
    if(dataset == "pubmedqa"):
        examples = pubmedqa_GIVE_a_prompt
    else:
        print("dataset undefined")
        exit()
    prompt = examples + "Q: "+ Q + "\n" + "Knowledge triplets: " + affirmative_knowledge
    messages = [
        {'role': 'system',
         'content':pubmedqa_GIVE_a_sys_msg},
        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.lower()
    return response

def GIVE_ac_answer(dataset, Q, affirmative_knowledge, answer_a, counterfactual_knowledge, model, temperature):
    if(dataset == "pubmedqa"):
        examples = pubmedqa_GIVE_ac_prompt
    else:
        print("dataset undefined")
        exit()
    prompt = examples + "Question: " + Q + "\n" + "Affirmative knowledge triplets: " + affirmative_knowledge + "\n" + "Previous answer: " + answer_a + "\n" + "Counter-factual knowledge triplets: " + counterfactual_knowledge
    messages = [
        {'role': 'system',
         'content':pubmedqa_GIVE_ac_sys_msg},
        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "Refined answer: {}"
        },

    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.lower()
    return response

def GIVE_ace_answer(dataset, Q, affirmative_knowledge, answer_a, counterfactual_knowledge, answer_ac, KG_knowledge, model, temperature):
    if(dataset == "pubmedqa"):
        examples = pubmedqa_GIVE_ace_prompt
    else:
        print("dataset undefined")
        exit()
    prompt = examples + "Question: " + Q + "\n" + "Affirmative knowledge triplets: " + affirmative_knowledge + "\n" + "Previous answer: " + answer_a + "\n" + "Counter-factual knowledge triplets: " + counterfactual_knowledge + "\n" + "Refined answer: " + answer_ac + "\n" + "Additional knowledge triplets from an expert knowledge base: " + KG_knowledge
    messages = [
        {'role': 'system',
         'content': pubmedqa_GIVE_ace_sys_msg},
        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "Further refined answer: {}"
        },

    ]
    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.lower()
    return response

def generate_answer_a_csqa(Q, choice, triplets, temperature, model):
    prompt = examples_10shot_commonsenseqa + """

            Q:""" + Q + """ Only return A, B, C, D or E""" + """"
            Choices:""" + choice + """
            Knowledge triplets: """ + triplets

    messages = [
        {'role': 'system',
         'content': """You are a helpful assistant that answers a given multiple choice question about common sense with A, B, C, D or E, based on the retrieved knowledge triplets (entity, relation, entity) and your own knowledge. """},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response


def generate_answer_ac_csqa(Q, choice, open_and_yes_txt, no_txt, answer_first, temperature, model):
    prompt = examples_10shot_commonsenseqa + """

            Q:""" + Q + """ Only return A, B, C, D or E""" + """
            Choices:""" + choice + """
            Knowledge triplets: """ + open_and_yes_txt + "\n" + "A:  " + answer_first + "\nAdditional knowledge triplets: " + no_txt
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given multiple choice question about common sense with A, B, C, D or E, based on the retrieved knowledge triplets (entity, relation, entity) and your own knowledge."},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response


def generate_answer_ace_csqa(Q, choice, open_and_yes_txt, no_txt, KG_txt, answer_first,
                                                answer_second, temperature,model):
    prompt = examples_10shot_commonsenseqa + """

            Q:""" + Q + """ Only return A, B, C, D or E""" + """
            Choices:""" + choice + """
            Knowledge triplets: """ + open_and_yes_txt + "\n" + "A:  " + answer_first + "\nAdditional knowledge triplets: " + no_txt + "\n" + "A: " + answer_second + "\n" + "Additional knowledge triplets retrieved from expert knowledge base: " + KG_txt

    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers a given multiple choice question about common sense with A, B, C, D or E, based on the retrieved knowledge triplets (entity, relation, entity) and your own knowledge"},

        {'role': 'user',
         'content': prompt
         },
        {
            'role': "assistant",
            'content': "{}"
        },

    ]

    response = get_completion_from_messages(messages, model=model, temperature=temperature)
    response = response.replace(".", "")
    response = response.replace(" ", "")
    index = response.find(":")
    if (index != -1):
        response = response[index + 1:]
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    response = response.capitalize()
    return response


