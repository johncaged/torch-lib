
def precision(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_positive)


def recall(true_positive, false_negative):
    if true_positive + false_negative == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_negative)
