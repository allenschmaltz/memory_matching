# this is part of the visualization code from the BLADE repo; not currently used for memmatch

import codecs

import numpy as np

def load_colors(filepath_with_name):
    color_gradients_pos_to_neg = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = list(line.strip())
            if len(line) == 6:
                color_gradients_pos_to_neg.append("#" + "".join(line))
    return color_gradients_pos_to_neg

def wrap_token_in_colored_span(token, color):
    """
    Wrap token in a span tag with a specified background color
    :param token: token string
    :param color: HTML HEX color code (e.g., #9C52B4)
    :return: token in a <span> tag with specified background color
    """
    return f'<span style="background-color: {color}; color: white;">{token}</span>'


def wrap_space_in_colored_span(color1, color2):
    """
    Wrap a space character in a span tag with a specified background gradient from color1 to color2
    :param color1: HTML HEX color code (e.g., #9C52B4)
    :param color2: HTML HEX color code (e.g., #9C52B4)
    :return: Space char (i.e., ' ') in a <span> tag with specified background gradient from color1 to color2
    """
    return f'<span style="background-image: linear-gradient(to right, {color1}, {color2});"> </span>'


def init_webpage_template():
    """
    Set the initial webpage template string
    :return: Valid webpage document HTML upto (and including) the initial <body> tag
    """
    webpage_template = """
     <!DOCTYPE html>
        <html>
        <head>
        <style>
        body {background-color: black; color: white;}
        del {
            background-color: #FC466B;
            }
        ins {
            text-decoration: underline;
            background-color: #3F5EFB;
            }
        </style>
        <title>CNN Visualization</title>
        </head>
        <body>
    """
    return webpage_template


def get_visualziation_lines_sentiment(test_y, original_untokenized_tokens, gold_lines, contribution_tuples_per_sentence, sentence_probs,
                            correct_color_code, wrong_color_code, detection_offset=0.0, fce_eval=True, viz_options=None):
    assert fce_eval
    webpage_template = ""
    assert len(test_y) == len(gold_lines)
    assert len(original_untokenized_tokens) == len(gold_lines) and len(gold_lines) == len(
        contribution_tuples_per_sentence) and len(contribution_tuples_per_sentence) == len(sentence_probs)
    sentence_index = 0
    for original_untokenized_sent, gold, generated, sentence_prob in zip(original_untokenized_tokens, gold_lines,
                                                                         contribution_tuples_per_sentence,
                                                                         sentence_probs):
        neg_sentence_prob = sentence_prob[0]
        pos_sentence_prob = sentence_prob[1]
        if pos_sentence_prob > neg_sentence_prob:
            y_softmax_pred = 1
        else:
            y_softmax_pred = 0
        if not fce_eval:
            gold_labels = convert_diffs_to_detection_labels(gold)
        else:
            gold_labels = [int(x) for x in gold]  # in this case, no final holder sym
        gold_viz_sentence = []
        predicted_viz_sentence = []
        assert len(original_untokenized_sent) == len(gold) and len(gold) == len(generated)
        predicted_color_codes = []
        gold_color_codes = []
        for gold_label, gradient_idx in zip(gold_labels, generated):
            neg_val, pos_val, neg_logit_bias, pos_logit_bias = gradient_idx
            if neg_val == 0.0 and pos_val == 0.0:
                # generated_labels.append(constants.ID_CORRECT)
                predicted_color_codes.append(correct_color_code)
            else:
                if pos_val + pos_logit_bias > neg_val + neg_logit_bias + detection_offset:
                    # generated_labels.append(constants.ID_WRONG)
                    predicted_color_codes.append(wrong_color_code)
                else:
                    # generated_labels.append(constants.ID_CORRECT)
                    predicted_color_codes.append(correct_color_code)

            if gold_label == 0:
                gold_color_codes.append(correct_color_code)
            else:
                gold_color_codes.append(wrong_color_code)

        for token_id in range(len(original_untokenized_sent)):
            next_token_id = np.minimum(token_id + 1, len(original_untokenized_sent) - 1)
            predicted_viz_sentence.append(
                wrap_token_in_colored_span(original_untokenized_sent[token_id], predicted_color_codes[token_id]))
            predicted_viz_sentence.append(
                wrap_space_in_colored_span(predicted_color_codes[token_id], predicted_color_codes[next_token_id]))

            gold_viz_sentence.append(
                wrap_token_in_colored_span(original_untokenized_sent[token_id], gold_color_codes[token_id]))
            gold_viz_sentence.append(
                wrap_space_in_colored_span(gold_color_codes[token_id], gold_color_codes[next_token_id]))

        if not viz_options["only_visualize_missed_predictions"] and not viz_options["only_visualize_correct_predictions"]:
            webpage_template += f"<h1>Sentence {sentence_index}: Softmax global predictions: Negative sentiment: {round(neg_sentence_prob, 2)}, Positive Sentiment: {round(pos_sentence_prob, 2)}</h1>"
            #webpage_template += f"<h1>Sentence {sentence_index}: Correct: {round(neg_sentence_prob, 2)}, Error: {round(pos_sentence_prob, 2)}</h1>"
            if test_y[sentence_index] != y_softmax_pred:
                if test_y[sentence_index] == 0:
                    webpage_template += f"<p>ERROR: GOLD: Negative sentiment</p>"  # this gets rid of the space
                elif test_y[sentence_index] == 1:
                    webpage_template += f"<p>ERROR: GOLD: Positive sentiment</p>"  # this gets rid of the space
                else:
                    assert False
            webpage_template += f"<p>GOLD: {''.join(gold_viz_sentence[0:-1])}</p>"  # this gets rid of the space
            webpage_template += f"<p>PRED: {''.join(predicted_viz_sentence[0:-1])}</p>"  # this gets rid of the space

            webpage_template += f"<hr>"
        elif viz_options["only_visualize_missed_predictions"]:
            if test_y[sentence_index] != y_softmax_pred:
                webpage_template += f"<h1>Sentence {sentence_index}: Softmax global predictions: Negative sentiment: {round(neg_sentence_prob, 2)}, Positive Sentiment: {round(pos_sentence_prob, 2)}</h1>"
                if test_y[sentence_index] == 0:
                    webpage_template += f"<p>ERROR: GOLD: Negative sentiment</p>"  # this gets rid of the space
                elif test_y[sentence_index] == 1:
                    webpage_template += f"<p>ERROR: GOLD: Positive sentiment</p>"  # this gets rid of the space
                else:
                    assert False
                webpage_template += f"<p>GOLD: {''.join(gold_viz_sentence[0:-1])}</p>"  # this gets rid of the space
                webpage_template += f"<p>PRED: {''.join(predicted_viz_sentence[0:-1])}</p>"  # this gets rid of the space

                webpage_template += f"<hr>"
        elif viz_options["only_visualize_correct_predictions"]:
            if test_y[sentence_index] == y_softmax_pred:
                webpage_template += f"<h1>Sentence {sentence_index}: Softmax global predictions: Negative sentiment: {round(neg_sentence_prob, 2)}, Positive Sentiment: {round(pos_sentence_prob, 2)}</h1>"
                if test_y[sentence_index] == 0:
                    webpage_template += f"<p>Correct prediction: GOLD: Negative sentiment</p>"  # this gets rid of the space
                elif test_y[sentence_index] == 1:
                    webpage_template += f"<p>Correct prediction: GOLD: Positive sentiment</p>"  # this gets rid of the space
                else:
                    assert False
                webpage_template += f"<p>GOLD: {''.join(gold_viz_sentence[0:-1])}</p>"  # this gets rid of the space
                webpage_template += f"<p>PRED: {''.join(predicted_viz_sentence[0:-1])}</p>"  # this gets rid of the space

                webpage_template += f"<hr>"
        sentence_index += 1
    return webpage_template