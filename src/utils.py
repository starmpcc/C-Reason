from evaluate import load

answer_phrase = "Therefore, the answer is "
response_template = "think step by step."


def simplify_string(i):
    delimiters = ["Q:", "\n\n", "</s>", "."]
    min_index = len(i)
    for delimiter in delimiters:
        index = i.find(delimiter)
        if 0 <= index < min_index:
            min_index = index
    return i[:min_index]


def get_accuracy(preds, labels):
    metric = load("exact_match")
    # corresponding to generate_until argument of harness
    preds = [simplify_string(i) for i in preds]

    acc = metric.compute(
        predictions=preds,
        references=labels,
        regexes_to_ignore=" ",
        ignore_case=True,
        ignore_punctuation=True,
    )

    return acc["exact_match"]
