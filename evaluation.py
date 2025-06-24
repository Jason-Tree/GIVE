from GIVE_functions import *
from sklearn.metrics import accuracy_score
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate for short-answer tasks")
    parser.add_argument('--dataset', type=str, default="pubmedqa",
                        help='Choose from {pubmedqa,bioasq,processbank,csqa}')
    parser.add_argument('--path', type=str, default = "GIVE_pubmedqa_a.json",
                        help='Please provide the path to the result json file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    pred_path = args.path
    predictions = json.load(open(pred_path))
    if(dataset == "pubmedqa"):
        ground_truth = json.load(open('data/QA/pubmedqa/test_ground_truth.json'))
        pmids = list(ground_truth)
        truth = [ground_truth[pmid] for pmid in pmids]
        preds = [predictions[pmid] for pmid in pmids]
        acc = accuracy_score(truth, preds)
        print('Accuracy %f' % acc)
    elif(dataset == "bioasq"):
        ground_truth = json.load(open("data/QA/BioASQ/keys.json"))
        num_correct = 0
        total = 0
        for key in list(predictions.keys()):
            total += 1
            correct = ground_truth[key]
            pred = predictions[key]
            if (correct == pred): num_correct += 1
        acc = num_correct / total
        print('Accuracy %f' % acc)
    elif(dataset == "processbank"):
        _,_,ground_truth = load_processbank()
        pred_list = list(predictions.values())
        num_correct = 0
        total = 0
        for idx in range(len(pred_list)):
            total += 1
            correct = ground_truth[idx]
            pred = pred_list[idx]
            if (correct == pred): num_correct += 1
        acc = num_correct / total
        print('Accuracy %f' % acc)

    elif(dataset == "csqa"):
        keys, ids, concepts, choices, questions = load_commonsenseqa()
        answers = json.load(open(args.path))
        c = 0
        for key in list((answers.keys())):
            a = answers[key]
            gt = keys[key]
            if (a == gt):
                c += 1
            acc = c / len(list(answers.keys()))
        print('Accuracy %f' % acc)
    else:
        print("Dataset undefined")
        exit()


