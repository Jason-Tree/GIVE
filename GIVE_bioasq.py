from GIVE_functions import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, default="",
                        help="your own openai api keys")
    parser.add_argument("--model_id", type = str, default="gpt-3.5-turbo",
                        help="backbone LLM model")
    parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",
                        help="sentence transformer model name from https://sbert.net/docs/sentence_transformer/pretrained_models.html")
    parser.add_argument("--temperature", type=int, default=0,
                        help="temperature for LLM output")
    parser.add_argument("--rewrite_question", type=bool, default=False,
                        help="whether or not to paraphrase the question statement")
    parser.add_argument("--entity_per_group", type=int, default=2,
                        help="number of KG entities introduced to each group")
    args = parser.parse_args()
    openai_key = args.openai_api_key
    G = pickle.load(open('data/KG/umls/umls_nx.pickle', 'rb'))
    openai.api_key = openai_key
    model = args.model_id
    sentence_transformer_id = args.sentence_transformer
    questions, answers = load_bioasq()
    answers_first = {}
    answers_second = {}
    answers_third = {}
    temperature = args.temperature
    entity_per_group = args.entity_per_group
    for key_idx in tqdm(range(len(questions))):
        key = answers[key_idx]
        Q = questions[key_idx]
        if(args.rewrite_question):
            Q_rewrite = rewrite_question(Q, model, temperature)
        else:
            Q_rewrite = Q

        entity_list = get_top_entities(Q_rewrite, model, temperature)
        relationship_list = get_top_relations(Q_rewrite, model, temperature)
        groups = []
        for entity in entity_list:
            groups.append(build_group(entity, entity_per_group, sentence_transformer_id))

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
                candidate_relation = add_relation_internal_knowledge(txt, model, temperature)
                if ('not related' in candidate_relation or 'unrelated' in candidate_relation):
                    continue
                else:
                    self_knowledge.append((group1[0], candidate_relation, entity))

            for j in range(i + 1, len(groups)):
                group2 = groups[j]
                yes_candidates, maybe_candidates, no_candidates, self_candidates, KG_candidates = build_connections(
                    Q_rewrite,
                    group1,
                    group2,
                    relationship_list,
                    model,
                    True,
                    sentence_transformer_id,
                    entity_per_group,
                    temperature)
                yes_knowledge.extend(yes_candidates)
                maybe_knowledge.extend(maybe_candidates)
                no_knowledge.extend(no_candidates)
                self_knowledge.extend(self_candidates)
                KG_knowledge.extend(KG_candidates)
        yes_knowledge = list(set(yes_knowledge))
        no_knowledge = list(set(no_knowledge))
        self_knowledge = list(set(self_knowledge))
        KG_knowledge = list(set(KG_knowledge))
        affirmative_knowledge = yes_knowledge + self_knowledge
        affirmative_txt = verbolize(affirmative_knowledge)
        counterfactual_txt = verbolize(no_knowledge)
        KG_txt = verbolize(KG_knowledge)
        answer_first = generate_answer_a_bioasq(Q_rewrite, affirmative_txt, temperature, model)
        if (counterfactual_txt != ""):
            answer_second = generate_answer_ac_bioasq(Q_rewrite, affirmative_txt, counterfactual_txt, answer_first,
                                                                     temperature, model)
        else:
            answer_second = answer_first
        if (KG_txt != ""):
            answer_third = generate_answer_ace_bioasq(Q_rewrite, affirmative_txt, counterfactual_txt, KG_txt, answer_first,
                                                                   answer_second, temperature, model)
        else:
            answer_third = answer_second
        answers_first[key_idx] = answer_first
        answers_second[key_idx] = answer_second
        answers_third[key_idx] = answer_third

    with open("GIVE_bioasq_a.json", "w") as outfile:
        json.dump(answers_first, outfile)
    with open("GIVE_bioasq_ac.json", "w") as outfile:
        json.dump(answers_second, outfile)
    with open("GIVE_bioasq_ace.json", "w") as outfile:
        json.dump(answers_third, outfile)