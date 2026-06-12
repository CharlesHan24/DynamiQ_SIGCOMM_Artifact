import torch
import pdb

def multiple_choice_dataset(examples, tokenizer):
    prompt = """Below is a multiple choice question. Select the correct answer among the given choices. You should answer one of A, B, C, or D.\n""" + "Question: {}\n" + "Choices:\n" + "A {}\nB {}\nC {}\nD {}\n" + "The correct answer is {}"

    choice_letters = ["A", "B", "C", "D"]
    
    questions = examples["question"]
    subjects = examples["subject"]
    choices_ = examples["choices"]
    answers = examples["answer"]
    res_input_ids = []
    res_attention_masks = []
    res_labels = []
    res_location = []
    res_choice = []
    PADDING_TOKEN_ID = -100
    
    for question, subject, choices, answer in zip(questions, subjects, choices_, answers):
        # Format the choices and answer according to the specified format
        if len(subject.split("_")) > 1:
            subject = " ".join(subject.split("_"))

        answer_text = choice_letters[int(answer)]
        text = prompt.format(question, *choices, answer_text)
        
        input_id = tokenizer(text,
                truncation='do_not_truncate',
                add_special_tokens=False
        ).input_ids

        attention_mask = [1] * len(input_id)
        labels = [PADDING_TOKEN_ID] * (len(input_id) - 1) + [input_id[-1]]

        res_input_ids.append(input_id)
        res_attention_masks.append(attention_mask)
        res_labels.append(labels)
        res_location.append(len(input_id) - 1)
        res_choice.append(answer)
    

    return {"input_ids": res_input_ids, "attention_mask": res_attention_masks, "labels": res_labels, "location": res_location, "res_choice": res_choice}