from GIVE_functions import *
def get_top_entities_commonsense(Q, choices, model):
    messages = [
        {'role': 'system',
         'content': """You are a helpful assistant that retrieves the top entities or concepts that contribute to a given multiple choice question about commonsense."""},
        {'role': 'user',
         'content': """Please retrieve the top entities or concepts that contribute to the given multiple choice question about common sense, seperated by comma.

                    Examples:

                    Question: Where are  you likely to find a hamburger?
                    Choices: A:fast food restaurant B:pizza C:ground up dead cows D:mouth E:cow carcus
                    Entities: {hamburger, fast food restaurant}.

                    Question: James was looking for a good place to buy farmland.  Where might he look?
                    Choices: A:midwest B:countryside C:estate D:farming areas E:illinois 
                    Entities: {farmland, midwest, countryside, farming areas, illinois}

                    Question: What do animals do when an enemy is approaching?
                    Choices: A:feel pleasure B:procreate C:pass water D:listen to each other E:sing 
                    Entities: {animals, procreate, pass water, listen to each other, sing}

                    Question: I did not need a servant.  I was not a what?
                    Choices: A:freedom B:rich person C:hired help D:in charge E:busy 
                    Entities: {servant, rich person}.

                    Question: An underrated thing about computers is how they manage workflow, at one time it was a big deal when they could first do what?
                    Choices: A:share files B:do arithmetic C:turn on D:cost money E:multitask 
                    Entities: {manage workflow, share files, do arithmatic, multitask} 

                    Question: James's nice asked him about her grandfather. She was interested in learning about what?
                    Choices: A:family tree B:family reunion C:babysitting D:brother's house E:heirlooms 
                    Entities: {family tree, family reunion, grandfather}.

                    Question: A beaver is know for building prowess, their supplies come from where?
                    Choices: A:british columbia B:body of water C:wooded area D:pay debts E:zoo 
                    Entities: {beaver, supplies, wooded area}.

                    Question: Zane doesn't like answering questions.  He's not good at it because he suffers from what?
                    Choices: A:panic B:discussion C:attention D:confusion E:satisfaction 
                    Entities: {answering questions, panic, confusion}

                    Question: The criminal insisted he must do the crime to the bank teller, but she tried to convince him there were other ways in life and this was what?
                    Choices: A:willing B:optional C:should not D:have to E:unnecessary 
                    Entities: {crime, optional, unnecessary}

                    Question: Going public about a common problem can gain what for a celebrity?
                    Choices: A:wide acceptance B:a degree C:pain D:getting high E:press coverage 
                    Entities: {goiong public, wide acceptance, press coverage}.

                    Question: Brawn opened the curtains so that the sun could do what?
                    Choices: A:dry clothes B:warm house C:warm room D:shine brightly E:get dark 
                    Entities:  {sun, dry cloths, warm house, warm room}

                    Given question:

                    Question: """ + Q + """
                    Choices: """ + choices
         },
        {
            'role': "assistant",
            'content': "Entities: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=0)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    entity_list = response.split(", ")
    for i in range(len(entity_list)):
        index = entity_list[i].find(":")
        if (index != -1):
            entity_list[i] = entity_list[i][index + 1:]
    return entity_list


def get_top_relations_commonsense(Q, choices, model):
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that retrieves the top relations or attributes that contribute to a given multiple-choice common sense question."},

        {'role': 'user',
         'content': """ Please retrieve the top relations or attributes that contribute to the given multiple-choice common sense question, seperated by comma. 

                        Examples:

                        Q: What do animals do when an enemy is approaching?
                        Choices: A:feel pleasure B:procreate C:pass water D:listen to each other E:sing 
                        Relationship or Attribute: {when enemy is approaching}.

                        Q: Reading newspaper one of many ways to practice your what?
                        Choices: A:literacy B:knowing how to read C:money D:buying E:money bank 
                        Relationship or attribute: {practice}.

                        Q: How would you get from one side of a canal to another?
                        Choices: A:michigan B:amsterdam C:venice D:bridge E:barges to travel on 
                        Relationship or attribute: {get from one side to another}.

                        Q: While washing clothes they became what when caught on the sharp object?
                        Choices: A:damaged B:wet clothes C:wear out D:torn E:have fun    
                        Relationship or attribute: {caught on the sharp object}.

                        Q: What would encourage someone to continue playing tennis?
                        Choices: A:becoming tired B:tennis elbow C:exercise D:hunger E:victory 
                        Relationship or attribute: {encourage}.

                        Q: A beaver is know for building prowess, their supplies come from where?
                        Choices: A:british columbia B:body of water C:wooded area D:pay debts E:zoo 
                        Relationship or attribute: {come from}.

                        Question: Zane doesn't like answering questions. He's not good at it because he suffers from what?
                        Choices: A:panic B:discussion C:attention D:confusion E:satisfaction 
                        Relationship or attribute: {suffer from}.

                        Question: The criminal insisted he must do the crime to the bank teller, but she tried to convince him there were other ways in life and this was what?
                        Choices: A:willing B:optional C:should not D:have to E:unnecessary 
                        Relationship or attribute: {is}.

                        Question: Joe found spiders while checking something outside. What might that be?
                        Choices: A:cupboard B:closet C:storage bag D:mail box E:garage 
                        Relationship or attribute: {might be outside}.

                        Given question:

                        Q: """ + Q + """
                        Choices: """ + choices
         },
        {
            'role': "assistant",
            'content': "Relationship or attribute: {}"

        }

    ]

    response = get_completion_from_messages(messages, model=model, temperature=0)
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

def add_relation_internal_knowledge_conceptnet(entities_txt, model):
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers the short relationship in a few words between two given common sense concepts."},

        {'role': 'user',
         'content': """Please answer a short relationship between two given common sense concepts using your own knowledge. 

                        Examples:

                        Q: victory, winning
                        A: {synonyms}.

                        Q: feeling good, fellpinion
                        A: {cause}.

                        Q: region, regional
                        A: {noun form of}.

                        Q: Great Britain, island country
                        A: {is an}.

                        Q: dog, apple
                        A: {unrelated}

                        Given concepts:
                        Q: """ + entities_txt},
        {
            'role': "assistant",
            'content': "A: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=0)
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    return response.lower()


def get_relation_between_groups_conceptnet(group1, group2, sparcity):
    if (sparcity == 1):
        G = pickle.load(open('data/KG/conceptnet/graph_full_new.gpickle', 'rb'))
    elif (sparcity == 0.1):
        G = pickle.load(open('data/KG/conceptnet/graph_10_percent_new.gpickle', 'rb'))
    elif (sparcity == 0.5):
        G = pickle.load(open('data/KG/conceptnet/graph_50_percent_new.gpickle', 'rb'))
    edges = [(u, d['relationship'], v) for u, v, d in G.edges.data() if u in group1 and v in group2]
    relations = [r for (e1, r, e2) in edges]
    return edges, relations


def query_if_relation_exists_conceptnet(entity1, relation, entity2, model):
    query = "(" + entity1 + ", " + relation + ", " + entity2 + ")"
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that answers yes,no or maybe depending on the correctness of the given triplet (entity, relation, entity) about commonsense."},

        {'role': 'user',
         'content': """Please answer yes, no or maybe about the correctness of the given commonsense triplet (concept, relation, concept).

                       Examples:

                       Triplet: (French people, french nationality, characteristic)
                       Correctness: {yes}

                       Triplet: (ethnic group, strong, passion for dancing)
                       Correctness: {no}

                       Triplet: (famous person, comes from, moon)
                       Correctness: {no}

                       Given Triplet:""" + query

         },
        {
            'role': "assistant",
            'content': "Correctness: {}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=0).lower()
    response.replace(".", "")
    response = response.lower()
    if (response[0] == 'y'):
        return "Y"
    elif (response[0] == 'n'):
        return "N"
    return "M"


def get_intermediate_node_conceptnet(group1, group2, Q, model, sparcity):
    if (sparcity == 1):
        G = pickle.load(open('data/KG/conceptnet/graph_full_new.gpickle', 'rb'))
    elif (sparcity == 0.1):
        G = pickle.load(open('data/KG/conceptnet/graph_10_percent_new.gpickle', 'rb'))
    elif (sparcity == 0.5):
        G = pickle.load(open('data/KG/conceptnet/graph_50_percent_new.gpickle', 'rb'))
    edges_out_group1 = [(u, d['relationship'], v) for u, v, d in G.edges.data() if u in group1]
    edges_in_group2 = [(u, d['relationship'], v) for u, v, d in G.edges.data() if v in group2]
    ziped_edges = list(zip(edges_out_group1, edges_in_group2))
    two_hops = [(a[0], a[1], b[0], b[1], b[2]) for a, b in ziped_edges if a[2] == b[0]]
    two_hops = [(a, b, c, d, e) for a, b, c, d, e in two_hops]
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
    if (len(knowledge_txt) < 2): return ""
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that selects one from the given 2-hop facts (entity, relaion, entity, relation, entity), that is most important to the given question. Return only the knowledge fact (entity, relaion, entity, relation, entity)."},

        {'role': 'user',
         'content': """Please select one from the given two-hop facts (entity, relation, entity, relation, entity), that is most important to the given common sense question.

                        Examples:

                        Question: Can nurse-led preoperative education reduce anxiety and postoperative complications in patients undergoing cardiac surgery?
                        Two-hop facts: (mental or behavioral dysfunction, precedes, experimental model of disease, complicates, injury or poisoning), (mental or behavioral dysfunction, complicates, experimental model of disease, occurs in, injury or poisoning), (mental or behavioral dysfunction, process of, experimental model of disease, co-occurs with, injury or poisoning), (mental or behavioral dysfunction, degree of, experimental model of disease, result of, injury or poisoning), (mental or behavioral dysfunction, result of, experimental model of disease, manifestation of, injury or poisoning), (mental or behavioral dysfunction, complicates, congenital abnormality, result of, injury or poisoning), (mental or behavioral dysfunction, co-occurs with, congenital abnormality, result of, therapeutic or preventive procedure)
                        Most important two-hop facts: {(mental or behavioral dysfunction, precedes, experimental model of disease, complicates, injury or poisoning)}

                        Given question about commonsense and the associated two-hop knowledge facts:

                        """ + knowledge_txt + "\n Question to answer: " + Q},
        {
            'role': "assistant",
            'content': "{}"
        }
    ]
    response = get_completion_from_messages(messages, model=model, temperature=0)
    index = response.find("(")
    if (index == -1): return ""
    index = response.find("{")
    if (index != -1):
        index2 = response.find("}")
        response = response[index + 1:index2]
    ans = response.split(",")
    if (len(ans) < 3): return ""
    ans = ans[2]
    if (ans[-1] == ")"): ans = ans[:-1]
    return ans.replace(" ", "")


def build_connections_conceptnet(query, group1, group2, query_relations, model, consider_intermediate,sparcity):
    """return all knowledge triplets from group1 to group2"""
    yes_candidates = []
    no_candidates = []
    maybe_candidates = []
    self_knowledge = []

    candidate_relations = []
    KG_knowledge, KG_relations = get_relation_between_groups_conceptnet(group1, group2, sparcity)
    candidate_relations.extend(KG_relations)
    candidate_relations.extend(query_relations)
    candidate_relations = list(set(candidate_relations))

    for i in range(len(group1)):
        for j in range(len(group2)):
            for relation in candidate_relations:
                self_knowledge_flag = query_if_relation_exists_conceptnet(group1[i], relation, group2[j], model)
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
            candidate_relation = add_relation_internal_knowledge_conceptnet(txt, model)
            if (
                    'not related' in candidate_relation or 'no direct relationship' in candidate_relation or 'unrelated' in candidate_relation):
                continue
            self_knowledge.append((group1[i], candidate_relation, group2[j]))
    if (consider_intermediate and len(yes_candidates) == 0):
        intermediate_node = get_intermediate_node_conceptnet(group1, group2, query, model, sparcity)
        if (intermediate_node != ""):
            intermediate_group = build_group_conceptnet(intermediate_node, G, all_embeddings, entity_per_group)
            yes_candidates1, maybe_candidates1, no_candidates1, self_knowledge1, KG_knowledge1 = build_connections_conceptnet(
                query, group1, intermediate_group, query_relations, model, False, sparcity)
            yes_candidates2, maybe_candidates2, no_candidates2, self_knowledge2, KG_knowledge2 = build_connections_conceptnet(
                query, intermediate_group, group2, query_relations, model, False, sparcity)
            yes_candidates.extend(yes_candidates1)
            yes_candidates.extend(yes_candidates2)
            maybe_candidates.extend(maybe_candidates1)
            maybe_candidates.extend(maybe_candidates2)
            no_candidates.extend(no_candidates1)
            no_candidates.extend(no_candidates2)
            self_knowledge.extend(self_knowledge1)
            self_knowledge.extend(self_knowledge2)
    return yes_candidates, maybe_candidates, no_candidates, self_knowledge, KG_knowledge

def rewrite_question_CSQA(Q, model, temperature):
    messages = [
        {'role': 'system',
         'content': "You are a helpful assistant that given a question about commonsense, that may contain two sentences, sub-sentence or passive voice, re-write it using one long question statement without passive voice."},
        {'role': 'user',
         'content': """Please re-write the question statement in one sentence without any sub-sentence or passive voice.

                        Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
                        Re-written question: {Where does a revolving door serve as a security measure, besides being convenient for two direction travel?}

                        Question:  I did not need a servant. I was not a what?
                        Re-written question: {Who needs a servant?}

                        Question: John and James are idiots. They bought two tickets to the Falcons vs the Jets even though neither wanted to see the what?
                        Re-written question: {What did neither idiots want to see even though they bought tickets to the Falcons vs the Jets?}


                        Question: """ + Q
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

def build_group_conceptnet(source_entity, G, all_embeddings, entity_per_group):
    group = [source_entity]
    landing_candidates = semantic_top_k_conceptnet([source_entity], entity_per_group, G, all_embeddings)
    for node_candidate in landing_candidates:
        node = node_candidate.replace("_", " ")
        node = node.lower()
        if (node in source_entity or source_entity in node):
            continue
        else:
            group.append(node)
    return list(set(group))

def semantic_top_k_conceptnet(Qs, k, G, concept_embeddings):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    query_embedding = encoder_model.encode(Qs, convert_to_tensor=True).to(device)
    concept_list = list(G)
    cosine_scores = util.cos_sim(query_embedding, concept_embeddings.to(device))
    cosine_scores = torch.sum(cosine_scores, 0)
    cosine_scores = torch.squeeze(cosine_scores)
    sorted_idx = torch.argsort(cosine_scores, descending=True)
    sorted_idx = sorted_idx.tolist()[:k]
    results = []
    for idx in sorted_idx:
        results.append(concept_list[idx].lower())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, default="",
                        help="your own openai api keys")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo",
                        help="backbone LLM model")
    parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",
                        help="sentence transformer model name from https://sbert.net/docs/sentence_transformer/pretrained_models.html")
    parser.add_argument("--temperature", type=int, default=0,
                        help="temperature for LLM output")
    parser.add_argument("--rewrite_question", type=bool, default=True,
                        help="whether or not to paraphrase the question statement")
    parser.add_argument("--entity_per_group", type=int, default=2,
                        help="number of KG entities introduced to each group")
    parser.add_argument("--sparcity_conceptnet", type=float, default=0.1,
                        help="sparcity of the ConceptNet KG for CSQA, choose from {1, 0.1, 0.5}")
    args = parser.parse_args()
    sparcity = args.sparcity_conceptnet
    if (sparcity == 1):
        G = pickle.load(open('data/KG/conceptnet/graph_full_new.gpickle', 'rb'))
        all_embeddings = torch.load("data/KG/conceptnet/conceptnet_1_all_embedding.pt")
    elif (sparcity == 0.1):
        G = pickle.load(open('data/KG/conceptnet/graph_10_percent_new.gpickle', 'rb'))
        all_embeddings = torch.load("data/KG/conceptnet/conceptnet_0.1_all_embedding.pt")
    elif (sparcity == 0.5):
        G = pickle.load(open('/home/jason/KG_LLM/data/KG/conceptnet/graph_50_percent_new.gpickle', 'rb'))
        all_embeddings = torch.load("data/KG/conceptnet/conceptnet_0.5_all_embedding.pt")
    else:
        print("Sparcity of ConceptNet undefined")

    entity_per_group = args.entity_per_group
    openai.api_key = args.openai_api_key
    model = args.model_id
    keys, ids, concepts, choices, questions = load_commonsenseqa()
    answers_first = {}
    answers_second = {}
    answers_third = {}
    temperature = args.temperature

    for question_idx in tqdm(range(len(questions))):
        id = ids[question_idx]
        concept = concepts[question_idx]
        choice = choices[question_idx]
        question = questions[question_idx]
        if(args.rewrite_question):
            question_rewrite = rewrite_question_CSQA(question, model, temperature)
        else:
            question_rewrite = question
        key = keys[id]
        entity_list = get_top_entities_commonsense(question_rewrite, choice, model)
        for entity in entity_list:
            if (concept in entity):
                entity_list.remove(entity)
        entity_list.append(concept)
        entity_list = list(set(entity_list))
        relationship_list = get_top_relations_commonsense(question_rewrite, choice, model)
        for entity in entity_list:
            for relation in relationship_list:
                if (entity in relation):
                    entity_list.remove(entity)

        groups = []
        for entity in entity_list:
            groups.append(list(set(build_group_conceptnet(entity, G, all_embeddings, entity_per_group))))

        yes_knowledge = []
        maybe_knowledge = []
        no_knowledge = []
        self_knowledge = []
        KG_knowledge = []
        for i in range(len(groups)):
            group1 = groups[i]
            for entity in group1[1:]:
                txt = ""
                txt += group1[0]
                txt += ", "
                txt += entity
                self_knowledge.append((group1[0], add_relation_internal_knowledge_conceptnet(txt, model), entity))

            for j in range(i + 1, len(groups)):
                group2 = groups[j]
                yes_candidates, maybe_candidates, no_candidates, self_candidates, KG_candidates = build_connections_conceptnet(
                    question_rewrite + " Choices:" + choice,
                    group1,
                    group2,
                    relationship_list,
                    model,
                    True,sparcity)
                yes_knowledge.extend(yes_candidates)
                maybe_knowledge.extend(maybe_candidates)
                no_knowledge.extend(no_candidates)
                self_knowledge.extend(self_candidates)
                KG_knowledge.extend(KG_candidates)
        yes_knowledge = list(set(yes_knowledge))
        no_knowledge = list(set(no_knowledge))
        self_knowledge = list(set(self_knowledge))
        KG_knowledge = list(set(KG_knowledge))

        knowledge_txt = ""

        open_and_yes = []
        open_and_yes.extend(yes_knowledge)
        open_and_yes.extend(self_knowledge)
        open_and_yes = [knowledge for knowledge in open_and_yes if knowledge[1] != ""]
        no_knowledge = [knowledge for knowledge in no_knowledge if knowledge[1] != ""]
        KG_knowledge = [knowledge for knowledge in KG_knowledge if knowledge[1] != ""]
        open_and_yes_txt = verbolize(open_and_yes)
        no_txt = verbolize(no_knowledge)
        KG_txt = verbolize(KG_knowledge)

        answer_first = generate_answer_a_csqa(question_rewrite, choice, open_and_yes_txt,
                                                                   temperature,
                                                                   model)
        if (no_txt != ""):
            answer_second = generate_answer_ac_csqa(question_rewrite, choice, open_and_yes_txt,
                                                                         no_txt,
                                                                         answer_first, temperature, model)
        else:
            answer_second = answer_first
        if (KG_txt != ""):
            answer_final = generate_answer_ace_csqa(question_rewrite, choice, open_and_yes_txt,
                                                                       no_txt, KG_txt,
                                                                       answer_first,
                                                                       answer_second, temperature, model)
        else:
            answer_final = answer_second
        answers_first[id] = answer_first
        answers_second[id] = answer_second
        answers_third[id] = answer_final

    with open("GIVE_csqa_a.json", "w") as outfile:
        json.dump(answers_first, outfile)
    with open("GIVE_csqa_ac.json", "w") as outfile:
        json.dump(answers_second, outfile)
    with open("GIVE_csqa_ace.json", "w") as outfile:
        json.dump(answers_third, outfile)



