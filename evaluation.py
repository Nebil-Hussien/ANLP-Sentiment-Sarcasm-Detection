from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from csv import DictWriter

def record_metrics(y_true, y_pred, file_name, display=False) -> tuple:
    '''
    saves, prints and returns f1 score, accuracy and confusion matrix
        y_true contains true labels
        y_pred contains predicted labels
        file_name should be file name without file extension or directory path
        display determines whether the computed values and plots are shown
        
    returns
        f1 score
        accuracy
        confusion matrix
    '''
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    svm_f1_diff = f1 - 0.275
    bert_f1_diff = f1 - 0.348
    
    heatmap = sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=['sincere', 'sarcastic'], yticklabels=['sincere', 'sarcastic'])
    plt.xlabel("true labels")
    plt.ylabel("predicted labels")
    if display:
        plt.show()
        print(f'The surveyed model {file_name} performs at an F1 score of {f1} and an accuracy of {acc}.')
        print(f'''It therefore outperforms the SemEval 2022
          SVM baseline by {svm_f1_diff} pp and the
          Bert baseline by {bert_f1_diff} pp.''')
    fig = heatmap.get_figure()
    fig.savefig(f"results/{file_name}_confusion.png")
    plt.close()
    
    with open(f"results/{file_name}_scores.csv", mode="w", encoding="utf-8-sig", newline="") as results:
        writer = DictWriter(results, fieldnames=["accuracy", "F1 score"])
        writer.writerow({"accuracy": "accuarcy", "F1 score": "F1 score"})
        writer.writerow({"accuracy": acc, "F1 score": f1})
    
    with open(f"results/{file_name}_preds.csv", mode="w", encoding="utf-8-sig", newline="") as preds:
        writer = DictWriter(preds, fieldnames=["correct"])
        writer.writeheader()
        for i in range(len(y_pred)):
            writer.writerow({"correct": y_true[i] == y_pred[i]})

    return f1, acc, mat

