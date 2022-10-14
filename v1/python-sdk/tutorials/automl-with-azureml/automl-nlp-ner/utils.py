from seqeval.metrics import accuracy_score
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
from azureml.automl.core.shared.constants import Metric


def named_entity_recognition_report(predictions_file, test_data_file, use_str=True):

    # getting the predicted and true labels
    def get_labels_from_file(file):

        labels = []
        current_labels = []
        with open(file) as opened:
            for line in opened.readlines()[1:]:

                if line == "-DOCSTART- O\n":
                    labels.append(current_labels)
                    current_labels = []

                elif line != "\n":
                    label = line.split()[1]
                    current_labels.append(label)

            labels.append(current_labels)
        return labels

    prediction_labels = get_labels_from_file(predictions_file)
    true_labels = get_labels_from_file(test_data_file)

    # accuracy
    accuracy = accuracy_score(y_true=true_labels, y_pred=prediction_labels)
    # micro averages
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true=true_labels, y_pred=prediction_labels, average="micro"
    )
    # macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true=true_labels, y_pred=prediction_labels, average="macro"
    )
    # weighted averages
    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_true=true_labels, y_pred=prediction_labels, average="weighted"
    )

    if use_str:
        results = Metric.Accuracy + "\t" + "{:.3f}".format(accuracy) + "\n"
        results += "\t\t" + "F1-score" + "\t" + "Precision" + "\t" + "Recall" + "\n"
        results += (
            "micro"
            + "\t\t"
            + "{:.3f}".format(f1_micro)
            + "\t\t"
            + "{:.3f}".format(precision_micro)
            + "\t\t"
            + "{:.3f}".format(recall_micro)
            + "\n"
        )
        results += (
            "macro"
            + "\t\t"
            + "{:.3f}".format(f1_macro)
            + "\t\t"
            + "{:.3f}".format(precision_macro)
            + "\t\t"
            + "{:.3f}".format(recall_macro)
            + "\n"
        )
        results += (
            "weighted"
            + "\t"
            + "{:.3f}".format(f1_weighted)
            + "\t\t"
            + "{:.3f}".format(precision_weighted)
            + "\t\t"
            + "{:.3f}".format(recall_weighted)
            + "\n"
        )

    else:
        results = dict()
        results[Metric.Accuracy] = accuracy
        results[Metric.F1Micro] = f1_micro
        results[Metric.F1Macro] = f1_macro
        results[Metric.F1Weighted] = f1_weighted
        results[Metric.PrecisionMicro] = precision_micro
        results[Metric.PrecisionMacro] = precision_macro
        results[Metric.PrecisionWeighted] = precision_weighted
        results[Metric.RecallMicro] = recall_micro
        results[Metric.RecallMacro] = recall_macro
        results[Metric.RecallWeighted] = recall_weighted

    return results
