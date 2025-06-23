rewrite_question_system_msg = "You are a helpful assistant that given a question that may contain sub-sentence or passive voice, re-write it using one long question statement without passive voice."
rewrite_question_prompts = """Please re-write the question statement in one sentence without any sub-sentence or passive voice.
Question: Ultrasound in squamous cell carcinoma of the penis; a useful addition to clinical staging?
Re-written question: {Is ultrasound a useful addition to clinical staging in cases of squamous cell carcinoma of the penis?}

Question: Learning needs of postpartum women: does socioeconomic status matter?
Re-written question: {Does socioeconomic status influence the learning needs of postpartum women?}

Question: Outcome Feedback within Emergency Medicine Training Programs: An Opportunity to Apply the Theory of Deliberate Practice?
Re-written question: {Is outcome feedback within Emergency Medicine Training Programs an opportunity to apply the Theory of Deliberate Practice?}

Question: """



get_entities_system_msg = "You are a helpful assistant that retrieves the top entities or concepts that contribute to a given biomedical question."
get_entities_prompts = """Please retrieve the top entities or concepts that contribute to the given biomedical question, seperated by comma.

Examples:

Question: Is ultrasound a useful addition to clinical staging in cases of squamous cell carcinoma of the penis?
Entities: {ultrasound, clinical staging, squamous cell carcinoma}.

Question: Does socioeconomic status influence the learning needs of postpartum women?
Entities: {socioeconomic status, learning needs, postpartum women}.

Question: Does every preschool child require specialized training in phonological awareness?
Entities: {preschool child, phonological awareness}

Question: Do scores from combining process indicators to evaluate the quality of care for surgical patients with colorectal cancer align with short-term outcomes?
Entities: {quality of care for surgical patients with colorectal cancer, short-term outcomes}

Given question:

Question: """

get_entities_processbank_system_msg = "You are a helpful assistant that retrieves the top entities or concepts that contribute to a given multiple choice biomedical question."
get_entities_processbank_prompts = """Please retrieve the top entities or concepts that contribute to the given multiple choice biomedical question, seperated by comma.

Examples:

Question: What is required for the wall to exert a back pressure on the cell?
Choices: A.an animal cell. B.water.
Entities: {back pressure, cell, water}

Question: What was the result of drift?
Choices: A.New alleles entered the population. B.the low egg-hatching rate.
Entities: {drift, new alleles, low egg-hatching rate}.
            
Question: Which two events can occur at the same time?
Choices: A.carbon fixation and the Calvin cycle. B.Carbon fixation and incorporating CO2 from the air into organic molecules.
Entities: {carbon fixation, Calvin cycle, incorporating CO2 from the air into organic molecules}.
                    
Question: reverse transcriptase does what?
Choices: A.makes cDNA. B.serves as a template.
Entities: {reverse transcriptase, cDNA, template}.  
                    
Question: The TEM uses what?
Choices: A.an electron beam. B.a light microscope.
Entities: {TEM, electron beam, light microscope}.

Question: Researchers focus on a region shared by what?
Choices:  A.affected people. B.unaffected people.
Entities: {researchers, affected people, unaffected people}.

Question: What is the direct action of enzymes?
Choices: A.export mRNA from the nucleus. B.modify the two ends of a eukaryotic pre-mRNA molecule.
Entities:{enzymes, mRNA, pre-mRNA molecule}
            
Given question:

"""

get_relations_system_msg = "You are a helpful assistant that retrieves the top relations or attributes that contribute to a given biomedical question."
get_relations_prompts = """Please retrieve the top relations or attributes that contribute to the given biomedical question, seperated by comma.

Examples:

Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
Relations or attributes: {play a role in}. The question asks if mitochondria play a role in programmed cell death, the relation that contributes to the question should be "play a role in". 

Question: Is ultrasound a useful addition to clinical staging in cases of squamous cell carcinoma of the penis?
Relationship or attributes: {a useful addition to}. The question asks if ultrasound is a useful addition to clinical staging, the relation that contributes to the question should be {a useful addition to}.

Question: Is the type of remission after a major depressive episode an important risk factor to relapses in a 4-year follow up?
Relationship or attributes: {an important risk factor to}. The question asks if the type of remission after a major depressive episode is an important risk factor in a 4-year follow up, the relation that contributes to the question should be {an important risk factor to}.

Question: Does every preschool child require specialized training in phonological awareness?
Relationship or attributes: {require specialized training in}

Question: Do scores from combining process indicators to evaluate the quality of care for surgical patients with colorectal cancer align with short-term outcomes?
Relationship or attributes: {align with}

Given question:

Question:"""

get_relations_processbank_system_msg = "You are a helpful assistant that retrieves the top relations or attributes that contribute to a given multiple-choice biomedical question."
get_relations_processbank_prompts = """Please retrieve the top relations or attributes that contribute to the given multiple-choice biomedical question, seperated by comma.

Examples:

Question: Which two events can occur at the same time?
Choices: A.carbon fixation and the Calvin cycle. B.Carbon fixation and incorporating CO2 from the air into organic molecules.
Relationship or attributes: {occur at the same time}.

Question: What would happen if genes were not transcribed?
Choices: A.Chains of amino acids would not be produced. B.genetic instructions would not be written in DNA.
Relationship or attributes: {were not transcribed, would not be produced, would not be written in}.
                        
Question: Researchers focus on a region shared by what?
Choices:  A.affected people. B.unaffected people.
Relationship or attributes: {focus on region shared by}

Question: What was the result of drift?
Choices: A.New alleles entered the population. B.the low egg-hatching rate.
Relationship: {the result of}

Given question:

"""

internal_knowledge_system_msg = "You are a helpful assistant that answers the short relationship in a few words between two given biomedical concepts."
internal_knowledge_prompts = """Please answer a short relationship between two given biomedical concepts using your own knowledge.

Examples:

Concepts: vitamin, neoplastic process
Relationship: {affects}.

Concepts:enzyme, cell function
Relationship: {disrupts}.

Concepts: cell component, entity
Relationship: {is a}.

Concepts: laboratory or test result, organ or tissue function
Relationship: {indicates}.

Concepts: animal, clinical staging
Relationship: {unrelated}.

Given concepts:

Concepts:"""


prune_candidate_knowledge_system_msg = "You are a helpful assistant that answers yes,no or maybe, depending on the correctness of the given biomedical triplet (concept, relation, concept)."
prune_candidate_knowledge_prompts = """Please answer yes, no or maybe about the correctness of the given biomedical triplet (concept, relation, concept).

Examples:

Triplet: (diagnostic arthroscopy, is a, tissue)
Correctness: {no}

Triplet: (medical device, is a, activity)
Correctness: {no}

Triplet: (diagnostic procedure, diagnoses, cell)
Correctness: {yes}

Triplet: (diagnostic procedure, diagnoses, cell)
Correctness: {yes}

Triplet: (ultrasound, diagnoses, neoplastic process)
Correctness: {yes}

Triplet: (ultrasound, measures, cell)
Correctness: {no}

Given triplet:

Triplet:"""

intermediate_node_system_msg = "You are a helpful assistant that selects one from the given two-hop knowledge facts (entity, relaion, entity, relation, entity), that is most important to the given biomedical question."
intermediate_node_system_prompts = """Please select one from the given two-hop facts (entity, relation, entity, relation, entity), that is most important to the given bio medical question.

Examples:

Question: Can nurse-led preoperative education reduce anxiety and postoperative complications in patients undergoing cardiac surgery?
Two-hop facts: (mental or behavioral dysfunction, precedes, experimental model of disease, complicates, injury or poisoning), (mental or behavioral dysfunction, complicates, experimental model of disease, occurs in, injury or poisoning), (mental or behavioral dysfunction, process of, experimental model of disease, co-occurs with, injury or poisoning), (mental or behavioral dysfunction, degree of, experimental model of disease, result of, injury or poisoning), (mental or behavioral dysfunction, result of, experimental model of disease, manifestation of, injury or poisoning), (mental or behavioral dysfunction, complicates, congenital abnormality, result of, injury or poisoning), (mental or behavioral dysfunction, co-occurs with, congenital abnormality, result of, therapeutic or preventive procedure)
Most important two-hop facts: {(mental or behavioral dysfunction, precedes, experimental model of disease, complicates, injury or poisoning)}

Given question about biomedical and the associated two-hop knowledge facts:

"""

examples_5shot_pubmedqa = """Q: Marital status, living arrangement and mortality: does the association vary by gender?
                       Knowledge Triplets: ('marital status', 'is a characteristic of', 'family group'),
                       ('family group', 'is related to', 'gender'),
                       ('living arrangement', 'not related', 'gender'),
                       ('living arrangement', 'influences', 'mortality'),
                       ('gender', 'characteristic of', 'age group'),
                       ('social behavior', 'influenced by', 'gender'),
                       ('mortality', 'influenced by', 'gender')
                       ('marital status', 'affects', 'mortality'), 
                       ('social behavior', 'influences', 'mortality')
                       ('marital status', 'influences', 'living arrangement')
                       ('social behavior', 'influences', 'living arrangement')
                       ('marital status', 'influences', 'social behavior'),
                       ('group', 'includes', 'gender')
                       Additional knowledge triplets: ('educational activity', 'not associated with', 'mortality')
                       Additional knowledge triplets retrieved from expert knowledge base: ('social behavior', 'associated with', 'age group')
                       A: Based on the given knowledge triplets, Marital status, living arrangement and mortality are all influnced by social behavior, and social behavior is influnced by gender, social behavior is also associated with age group. of which gender is a characteristic of. Although living arrangement is not directly related to gender, we can infer that the association between them vary by gender from other knowledge, so the correct answer to this question should be {yes}. 

                       Q：Do patients with rheumatoid arthritis established on methotrexate and folic acid 5 mg daily need to continue folic acid supplements long term?
                       Knowledge Triplets: ('methotrexate', 'need to continue', 'folic acid supplements')
                       ('methotrexate', 'interacts', 'folic acid supplements'),
                       ('disease or syndrome', 'benefits from', 'folic acid supplements'),
                       ('rheumatoid arthritis', 'affects', 'folic acid supplements')
                       Additional knowledge triplets:  ('rheumatoid arthritis', 'not produces', 'folic acid supplements')
                       Additional knowledge triplets retrieved from expert knowledge base: ('pharmacologic substance', 'interacts with', 'vitamin')
                       A: Based on the given knowledge triplets, rheumatoid arthritis affect and does not produces folic acid supplements, disease or syndrome benefits from folic acid supplements, and methotrexate interacts with folic acid supplements, so the correct answer to this question should be {yes}.

                       Q: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
                       Knowledge Triplets: ('mitochondria', 'affects', 'programmed cell death'),
                       ('mitochondria', 'play a role in', 'remodelling'),
                       ('mitochondria', 'affects', 'plant'),
                       Additional knowledge triplets: ('mitochondria', 'not location of', 'programmed cell death'), 
                       ('mitochondria', 'not result of', 'cell or molecular dysfunction'),
                       ('mitochondria', 'play a role in', 'programmed cell death')
                       Additional knowledge triplets retrieved from expert knowledge base: ('cell', 'part of', 'plant'), 
                       ('organism function', 'result of', 'cell or molecular dysfunction'), 
                       ('cell', 'location of', 'cell or molecular dysfunction'), 
                       ('organism function', 'process of', 'cell or molecular dysfunction'), 
                       ('organism function', 'affects', 'cell or molecular dysfunction')
                       A: Based on the given knowledge triplets, mitochondria affects and plays a role in programmed cell death, the correct answer to this question should be {yes}.

                       Q: Combining process indicators to evaluate quality of care for surgical patients with colorectal cancer: are scores consistent with short-term outcome?
                       Knowledge triplets: ('scores', 'measures', 'phenomenon or process'),
                       ('process indicators', 'related', 'short-term outcome'),
                       ('quality of care', 'affects', 'scores'),
                       ('patient or disabled group', 'affects', 'short-term outcome')
                       ('natural phenomenon or process', 'affects', 'patient or disabled group')
                       Additional knowledge triplets: ('quality of care', 'not manages', 'patient or disabled group'),  
                       ('quality of care', 'not is a', 'scores')
                       Additional knowledge triplets retrieved from expert knowledge base:('natural phenomenon or process', 'process of', 'neoplastic process'),
                       ('health care activity', 'affects', 'neoplastic process')
                       A: Based on the knowledge triplets, quality of care affects scores but is not scores, so higher scores dont mean better quality of care. Natural phenomenon or process
                       affects patient ot disabled group, which further affects short-term outcome, and healthcare activity also affects neoplastic process, so quality of care and its scores
                       are not necessarily consistent with short-term outcome, as it is affected by multiple factors. Therefore, the correct answer to this question should be {maybe}.

                       Q: Can nonproliferative breast disease and proliferative breast disease without atypia be distinguished by fine-needle aspiration cytology?
                       Knowledge triplets: ('fine-needle aspiration cytology', 'diagnoses', 'nonproliferative breast disease'), 
                       ('fine-needle aspiration cytology', 'diagnoses', 'disease or syndrome'),
                        ('fine-needle aspiration cytology', 'is a', 'molecular biology research technique')
                       Additional knowledge triplets: ('fine-needle aspiration cytology', 'not measures', 'proliferative breast disease')
                       Additional knowledge triplets retrieved from expert knowledge base:  ('molecular biology research technique', 'measures', 'disease or syndrome'), 
                       ('laboratory procedure', 'measures', 'disease or syndrome')
                       A: Based on the knowledge triplets, fine-needle aspiration cytology diagnoses nonproliferative breast disease but doesnot measure proliferative breast disease. Therefore, the correct answer to this question should be {no}.

"""

examples_5shot_bioasq = """   Q: Can valproic acid act as an activator of AMPK?
                       Knowledge Triplets: ('valproic acid', 'is a', 'chemical viewed functionally'),
                       ('molecular function', 'related_to', 'AMPK'),
                       ('valproic acid', 'affects', 'chemical viewed functionally'),
                       ('chemical viewed functionally', 'regulates', 'AMPK')
                       Additional knowledge triplets: ('valproic acid', 'not act as an activator of', 'alga'),
                       ('organophosphorus compound', 'not affects', 'molecular sequence')
                       Additional knowledge triplets retrieved from expert knowledge base: 
                       Based on the given knowledge triplets, valproic acid is a chemical viewed functionally, which regulates AMPK, so the correct answer to this question should be {yes}. 


                       Q：Does Chromatin Immunoprecipitation (ChIP) show a bias for highly expressed loci?
                       Knowledge Triplets: ('Chromatin Immunoprecipitation (ChIP)', 'identifies', 'highly expressed loci')，
                        ('Chromatin Immunoprecipitation (ChIP)', 'analyzes', 'molecular sequence')，
                       Additional knowledge triplets: ('Chromatin Immunoprecipitation (ChIP)', 'not interacts with', 'inorganic chemical')
                       Additional knowledge triplets retrieved from expert knowledge base: ('nucleic acid nucleoside or nucleotide', 'interacts with', 'inorganic chemical')
                       A: Based on the given knowledge triplets, Chromatin Immunoprecipitation (ChIP) identifies highly expressed loci, so the correct answer to this question should be {yes}.

                       Q: Are thyroid hormone receptors implicated in arterial hypertension?
                       Knowledge Triplets: ('thyroid hormone receptors', 'affects', 'physiologic function'),
                        ('steroid', 'affects', 'physiologic function'),
                        ('thyroid hormone receptors', 'implicated in', 'physiologic function'),
                        ('arterial hypertension', 'affects', 'physiologic function')
                       Additional knowledge triplets: ('thyroid hormone receptors', 'not ingredient of', 'arterial hypertension'), 
                       ('thyroid hormone receptors', 'not disrupts', 'physiologic function')
                       Additional knowledge triplets retrieved from expert knowledge base: ('hormone', 'disrupts', 'physiologic function'), 
                       ('steroid', 'affects', 'physiologic function')
                       A: Based on the given knowledge triplets, although thyroid hormone receptors affects physiologic function, it is not ingredient of arterial hypertension, so it is not implicated in arterial hypertension, the correct answer to this question should be {no}.

                        Q: Is there an association between borna virus and brain tumor?
                        Knowledge Triplets: ('borna virus', 'there is no known relationship between borna virus and rickettsia or chlamydia.', 'rickettsia or chlamydia'),
                        ('rickettsia or chlamydia', 'causes', 'brain tumor')
                       Additional knowledge triplets: ('borna virus', 'not causes', 'neoplastic process'),
                       ('rickettsia or chlamydia', 'not association', 'brain tumor'),
                       ('virus', 'not causes', 'brain tumor'),
                       ('borna virus', 'not association', 'brain tumor')
                       Additional knowledge triplets retrieved from expert knowledge base: ('rickettsia or chlamydia', 'causes', 'neoplastic process'), 
                       ('virus', 'causes', 'neoplastic process'), 
                       ('bacterium', 'causes', 'neoplastic process')
                       A: Based on the given knowledge triplets, although ricjettsia or chlamydia causes brain tumor, borna virus is not related to them. Besides, borna virus does not cause neoplastic process and is not associated with brain tumor, so the correct answer to this question should be {no}.

                        Q: Is endostatin a proangiogenic factor?
                        Knowledge Triplets: ('endostatin', 'disrupts', 'proangiogenic factor'),
                        ('endostatin', 'inhibits', 'steroid'),
                        ('endostatin', 'regulates', 'immunologic factor'),
                        ('endostatin', 'regulates', 'hormone'),
                        ('steroid', 'affects', 'proangiogenic factor'),
                         ('hormone', 'regulates', 'immunologic factor')
                       Additional knowledge triplets: ('endostatin', 'not causes', 'proangiogenicfactor')
                       Additional knowledge triplets retrieved from expert knowledge base: ('steroid', 'interacts with', 'immunologic factor'), 
                       ('hormone', 'disrupts', 'genetic function'), 
                       ('hormone', 'affects', 'genetic function'), 
                       ('steroid', 'affects', 'genetic function')
                       A: Based on the given knowledge triplets, endostatin disrupts proangiogenic factor, endostatin inhibits steroid and regulates hormone, both of them affect or regulate proangiogenic factor, so the correct answer to this question should be {no}.
                        """

examples_10shot_processbank = """
    Q: What was required for the evolution of greater morphological diversity?
    Choices: A.simpler prokaryotic cells. B.the first eukaryotes.
    Knowledge Triplets: (evolution, influences, organism),
    (evolution, related, eukaryotes),
    (organism, isa, eukaryotes),
    (evolution, related, eukaryotes),
    (anatomical structure, part of, eukaryotes),
    (anatomical structure, part of, organism)
    Additional Knowledge Triplets: (anatomical structure, not related to, prokaryotic cells),
    Additional knowledge triplets retrieved from expert knowledge base: (anatomical structure, part of, organism)
    A: Based on the retrieved knowledge triplets, evolution influences organism and is related to eukaryotes, anatomical structure is part of organism and eukaryotes but not related to prokaryotic cells, so the correct answer to this question is {B}.


    Q: Small proteins are attached to ubiquitin
    Choices: A.False. B.True.
    Knowledge Triplets: (amino acid peptide or protein, involved in degradation, ubiquitin), 
    (amino acid peptide or protein, involved in, eicosanoid), 
    (Small proteins, tagged by, ubiquitin), 
    (Small proteins, a type of, amino acid peptide or protein)
    A: Based on the retrieved knowledge triplets, small proteins involved in degradation of ubiquitin and are tagged by ubiquitin, so the statement that small proteins are attached to ubiquitin is false, the correct answer to this question is {A}.

    Q: Allopatric speciation occurs when gene flow to and from the isolated population is blocked.
    Choices: A.True. B.False.
    Knowledge Triplets: (Allopatric speciation, blocked, gene flow), 
    (Allopatric speciation, occurs when, isolated population), 
    (Allopatric speciation, occurs when, gene flow), 
    (gene or genome, present in, isolated population), 
    (Allopatric speciation, involves, population group), 
    (gene flow, related, population group), 
    (gene flow, affects, isolated population), 
    (Allopatric speciation, results in, isolated population), 
    (isolated population, subset, population group), 
    Additional knowledge triplets:(organism, not occurs when, population group), 
    (organism, not blocked, population group)
    A: Based on the given knowledge triplets, Allopatric speciation occurs with isolated populationand gene flow, so when the gene flow is blocked, Allopatric speciation should not occur, the statement if false, the correct answer to this question is {B}.

    Q: Polysaccharide agarose does what?
    Choices: A.studies DNA molecules. B.acts as a molecular sieve.
    Knowledge Triplets: (Polysaccharide agarose, affects, DNA molecules),
    (Polysaccharide agarose, is a, molecular sieve),
    (Polysaccharide agarose, affects, nucleic acid nucleoside or nucleotide),
    (Polysaccharide agarose, isa, carbohydrate),
    (Polysaccharide agarose, isa, chemical),
    (carbohydrate, is_a, molecular sieve)
    (Polysaccharide agarose, substrate, enzyme)
    Additional knowledge Triplets: (carbohydrate, not complicates, DNA molecules),
    (carbohydrate, not interacts with, molecular sieve),
    (enzyme, not complicates, molecular sequence),
    (enzyme, not complicates, molecular sieve)
    Additional knowledge triplets retrieved from expert knowledge base: (enzyme, affects, molecular function), 
    (enzyme, complicates, molecular function),
    (enzyme, disrupts, molecular function), 
    (carbohydrate, interacts with, chemical)
    A: Based on the given knowledge triplets, Polysaccharide agarose is a molecular sieve. Polysaccharide agarose is a carbohydrate, which is a molecular sieve. Also, Polysaccharide agarose substrates enzyme, which affects, complicates and disrupts molecular function. Therefore, the correct choice to this question should be {B}.

    Q: Which two events can occur at the same time?
    Choices: A.carbon fixation and the Calvin cycle. B.Carbon fixation and incorporating CO2 from the air into organic molecules.
    Knowledge Triplets: (carbon fixation, interacts with, incorporating CO2 from the air into organic molecules), 
    (Calvin cycle, complicates, incorporating CO2 from the air into organic molecules),
    (carbon fixation, interacts with, plant),
    (plant, affects, incorporating CO2 from the air into organic molecules),
    (incorporating CO2 from the air into organic molecules, involves, molecular function),
    (organic chemical, affects, molecular function),
    (carbon fixation, incorporating, incorporating CO2 from the air into organic molecules),
    (carbon fixation, part of, carbohydrate sequence),
    (carbohydrate sequence, related to, chemical viewed functionally), 
    (carbohydrate, incorporates, incorporating CO2 from the air into organic molecules)
    Additional Knowledge Triplets: (carbohydrate sequence, not disrupts, organic chemical),
    (carbohydrate sequence, not disrupts, molecular function)
    Additional knowledge triplets retrieved from expert knowledge base: (carbohydrate, affects, molecular function)
    A: Based on the given knowledge triplets, carbon fixation incorporating and interacts with incorporating CO2 from the air into organic molecules. Besides, carbon fixation is part of carbohydrate sequence, carbohydrate incorporate incorporating CO2 from the air into organic molecules. Also, carbon fixation interacts with, plant,
    and plan affects incorporating CO2 from the air into organic molecules. Therefore, the correct choice to this question should be {B}.

    Q: What was the result of drift?
    Choices: A.New alleles entered the population. B.the low egg-hatching rate.
    Knowledge Triplets: (disease or syndrome, causes, low egg-hatching rate)，
    (drift, associated with, low egg-hatching rate)， 
    (drift, causes, acquired abnormality)，
    (drift, associated with, disease or syndrome)，
    (drift, affects, genetic function)，
    (genetic function, influences, embryonic structure)，
    (congenital abnormality, causes, low egg-hatching rate)，
    (congenital abnormality, is_a, embryonic structure)，
    (genetic function, affects, low egg-hatching rate)
    (acquired abnormality, leads to, low egg-hatching rate)
    Additional Knowledge Triplets:  (drift, not causes, congenital abnormality),
    (new alleles, not the result of, embryonic structure)
    Additional Knowledge Triplets retrieved from expert knowledge base: (genetic function, affects, invertebrate),
    (congenital abnormality, part of, invertebrate)
    A: Based on the given knowledge triplets, drift is associated with low egg-hatching rate. Drift causes acquired abnormality, which leads to low egg-hatching rate. Drift also affects genetic function, which affects low egg-hatching rate. Therefore, the correct choice to this question should be {B}.

    Q: Which event occurs first?
    Choices: A.a clone of cells is formed. B.plating out all the bacteria.
    Knowledge Triplets: (clone of cells, affected by, antibiotic),
    (cell, interacts, bacterium),
    (plating out all the bacteria, involves, bacterium),
    (cell function, related, plating out all the bacteria)
    Additional Knowledge Triplets: 
    Additional Knowledge Triplets retrieved from expert knowledge base: (cell, part of, bacterium), 
    (cell component, location of, bacterium)
    A: Based on the given knowledge triplets, cell interacts bacterium, cell function is related to plating out all the bacteria, so plating out all the bacteria should occur first. The correct choice to this question should be {B}.

    Q: What causes one or more extra sets of chromosomes?
    Choices: A.polyploidy. B.an accident in meiosis.
    Knowledge Triplets: (extra sets of chromosomes, isa, congenital abnormality),
    (extra sets of chromosomes, result of, accident in meiosis),
    (gene or genome, associated with, accident in meiosis),
    (extra sets of chromosomes, affects, gene or genome),
    (genetic function, involved in, accident in meiosis),
    (accident in meiosis, leads to, congenital abnormality)
    Additional Knowledge Triplets: 
    Additional Knowledge Triplets retrieved from expert knowledge base: (gene or genome, location of, injury or poisoning), 
    (gene or genome, location of, acquired abnormality),
    (genetic function, affects, organism), 
    (gene or genome, part of, organism)
    A: Based on the given knowledge triplets, accident in meiosis leads to congenital abnormality, and extra sets of chromosomes is a congenital abnormality. Also, extra sets of chromosomes is result of accident in meiosis. The correct choice to this question should be {B}.

    Q: DNA ligase does what?
    Choices: A.makes associations permanent. B.cleaves the sugar-phosphate backbones in the two DNA strands.
    Knowledge Triplets:  (DNA ligase, interacts with, enzyme),
    (gene or genome, cleaves, cleaves the sugar-phosphate backbones),
    (DNA ligase, joins, nucleotide sequence),
    (DNA ligase, is a, enzyme),
    (nucleotide sequence, related to, enzyme),
    (nucleotide sequence, cleaves, cleaves the sugar-phosphate backbones)
    Additional Knowledge Triplets: (nucleotide sequence, not does, makes associations permanent)
    Additional Knowledge Triplets retrieved from expert knowledge base: (nucleic acid nucleoside or nucleotide, interacts with, enzyme),
    A: Based on the given knowledge triplets, DNA ligase is a enzyme, which cleaves the sugar-phosphate backbones. The correct choice to this question should be {B}.

    Q: What would happen without the specimen?
    Choices: A.the image of the specimen would not be projected. B.lenses would not refract the light.
    Knowledge Triplets: (specimen, represented by, image),
    (specimen, not related to, lenses),
    (specimen, not related to, light)
    Additional Knowledge Triplets:
    Additional Knowledge Triplets retrieved from expert knowledge base:
    A: Based on the given knowledge triplets, specimen is represented by image, so without specimen, the image of the specimen would not be projected. The correct choice to this question should be {A}.

"""

examples_10shot_commonsenseqa = """
    Q: Where do you find the most amount of leafs?
    Choices: A:floral arrangement B:ground C:forrest D:field E:compost pile 
    Knowledge Triplets: (leaves, fall on, ground),
    (leaf, location, ground),
    (leaf, part-whole, forest),
    (leaves, location, forest),
    (ground, part of, forest)
    Based on the retrieved knowledge triplets, leaves are on the ground and forest, but ground is also a part of forest. Therefore, the correct answer to this question is {C}.

    Q: What is happening while he's playing basketball for such a long time?
    Choices: A:sweating B:pain C:having fun D:medium E:knee injury 
    Knowledge Triplets: (basketballs, can involve, enjoying),
    (basketballing, activity and enjoyment, having fun),
    (playing basketball, activity and feeling, enjoying),
    (basketballs, associated with, sweating),
    (enjoying, antonyms, sweating)
    Based on the retrieved knowledge triplets, playing basketball is related to having fun and sweating. However, the question asks what is happening, the answer must be a specific event, which is sweating. Therefore, the correct answer to this question is {A}.


    Q: If you're going to a party in a new town what are you hoping to make?
    Choices: A:getting drunk B:making new friends C:new contacts D:doing drugs E:set home 
    Knowledge Triplets: (first contacts, initial stage of, make friends), 
    (going to party, opportunity for, make friends),
    (party to, forming relationships, making friends),
    (party to, leads to, new contacts),
    (new contacts, can lead to, friendships),
    (new contacts, forming relationships, making friends),
    (new contacts, process of, make friends)
    Based on the retrieved knowledge triplets, going to party helps making friends and new contacts. However, new contacts also lead to friendships and making friends. Therefore, the correct answer to this question should be {B}.

    Q: Brawn opened the curtains so that the sun could do what?
    Choices: A:dry clothes B:warm house C:warm room D:shine brightly E:get dark 
    Knowledge Triplets: (sun, provides, warmate),
    (comfort room, provides, warmness),
    (warm house, part of, warm room),
    (warm house, provides comfort, comfort room),
    (sun, causes, warm room)
    Based on the retrieved knowledge triplets, sun provides warmate and causes warm room, warm house is also part of warm room. Therefore, the correct answer to this question should be {C}.

    Q: What island country is ferret popular?
    Choices: A:own home B:north carolina C:great britain D:hutch E:outdoors 
    Knowledge Triplets: (Great Britain, is an, island country),
    (britain, location, island)
    Additional Knowledge Triplets:
    Additional knowledge triplets retrieved from expert knowledge base:
    Based on the retrieved knowledge triplets, Great Britain is an island country. The question asks which island country is ferret popular, so the answer must be an island country, the only choice is great britain. Therefore, the correct choice to this question should be {C}.

    Q: What do people typically do while playing guitar?
    Choices: A:cry B:hear sounds C:singing D:arthritis E:making music 
    Knowledge Triplets: (playing guitar, musical activities, singing),
    (sing, part of, making music),
    (classical_guitarists, performing, singing),
    (playing guitar, musical activities, sing),
    (singing, part of, making music),
    (singing, part of, music)
    Additional Knowledge Triplets:
    Additional knowledge triplets retrieved from expert knowledge base:
    Based on the retrieved knowledge, playing guitar and singing are musical activities, singing is also a part of making music. Besides, the question asks what people typically do, so the answer must be a common activity of people, making music is not a common activity. Therefore, the correct choice to this question should be {C}.

    Q: I did not need a servant. I was not a what?
    Choices: A:freedom B:rich person C:hired help D:in charge E:busy 
    Knowledge Triplets:  (maid, employment, rich person),
    (servant, employer-employee, rich person),
    (even_servant, opposite, rich person),
    (maid, employed by, wealthy),
    (servant, employed by, wealthy)
    Additional Knowledge Triplets:
    Additional knowledge triplets retrieved from expert knowledge base:
    Based on the retrieved knowledge, servant is employed by wealthy rich person, so if I did not need a servant, I was not a rich person. Therefore, the correct choice to this question should be {B}.

    Q: An underrated thing about computers is how they manage workflow, at one time it was a big deal when they could first do what?
    Choices: A:share files B:do arithmetic C:turn on D:cost money E:multitask 
    Knowledge Triplets: (multitask, capability, computer),
    (multitask, skill and task, manage workflow),
    (multitask, involves, manage),
    (multitask, involves, multisentence),
    (taskmaster, related to, manage workflow),
    (multitask, part of, workflow)
    Additional Knowledge Triplets: (computer, not first do, manage workflow),
     (multisentence, not first do, manage workflow)
    Additional knowledge triplets retrieved from expert knowledge base:
    Based on the retrieved knowledge, manage workflow is the skill and task of multitask, so manage workflow is a big deal when computers can first do multitask. Therefore, the correct choice to this question should be {E}.

    Q: Obstructing justice is sometimes an excuse used for police brutality which causes what in people?
    Choices: A:committing perjury B:prosecution C:attack D:getting hurt E:riot 
    Knowledge Triplets: (police brutality, involves, hurting),
    (riot, can be the result of, brutality),
    (brutality, opposites, justice),
    (brutality, actions, hurting),
    (police brutality, can result in, postinjury),
    (police brutality, cause and effect, getting hurt),
    (brutality, result of, injury), 
    (police brutality, cause, injury),
    (riot, can involve, hurting),
    (riot, cause, pain),
    (riot, can result in, getting hurt)
    Additional Knowledge Triplets: (riot, not excuse for police brutality, brutality).
    Additional knowledge triplets retrieved from expert knowledge base: (postinjury, RelatedTo, injury)
    Based on the retrieved knowledge, police brutality involves and can result in hurting, it causes and effects getting hurt. Besides, police brutality can result in postinjury, which is related to injury. Lastly, brutality can result in riot, which also results in pain and getting hurt. Therefore, the correct choice to this question should be {D}.

    Q: Going public about a common problem can gain what for a celebrity?
    Choices: A:wide acceptance B:a degree C:pain D:getting high E:press coverage 
    Knowledge Triplets: (publically, adverb form of, acceptances)，
    (going public, cause and effect, wide acceptance),
    (press coverage, cause and effect, wide acceptance),
    (publicizes, cause, wide acceptance),
    (going public, leads to, acceptances)
    Based on the retrieved knowledge, going public cause and effect wide acceptance. Besides, press coverage also cause and effect wide acceptance. Therefore, the correct answer to this question should be {D}. 
"""




















