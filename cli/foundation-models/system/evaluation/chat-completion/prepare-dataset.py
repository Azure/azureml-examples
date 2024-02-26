import argparse
import pandas as pd
from datasets import load_dataset

test_data = "./small-test.jsonl"

test_data_values = [
    {
        "prompt": [
            {
                "role": "user",
                "content": "### Question: A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, \"Listen, you bastard, I'll wear this rag anywhere I please. \" The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that\nA. the statute is void for vagueness under the Fourteenth Amendment's due process clause.\nB. the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.\nC. the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.\nD. the statute is overbroad and consequently invalid under the First and FourteenthAmendments.\n\n### Answer: ",
            },
            {"role": "assistant", "content": "D"},
            {
                "role": "user",
                "content": "### Question: One Sunday afternoon, a sports fan went to a professional football game. While standing in line at a concession stand, he got into an argument with three men ahead of him in the line. The three men were friends attending the game together. One of the men punched the sports fan in the mouth. A fight ensued. During the fracas, one of the other three men hit the sports fan over the head with a bottle, which caused him to suffer a serious concussion. The sports fan does not know for certain which of the three men struck him with the bottle. If the sports fan asserts a claim against the man who punched him in the mouth to recover damages for the head injury, will the sports fan prevail?\nA. Yes, because the man who punched him in the mouth was the instigator who struck the first blow that started the fracas.\nB. Yes, because the man who punched him in the mouth and the other men were acting in concert.\nC. No, because the man who punched him in the mouth did not actually strike the sports fan with the bottle.\nD. No, because the sports fan cannot offer proof as to which of the men struck the sports fan over the head with the bottle.\n\n### Answer: ",
            },
        ],
        "ground_truth": "B",
        "prompt_length": 468,
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "### Question: Which word best summarizes Weber's explanation of the development of formally rational law?\nA. Authority.\nB. Charisma.\nC. Co-operation.\nD. Capitalism.\n\n### Answer: ",
            },
            {"role": "assistant", "content": "D"},
            {
                "role": "user",
                "content": '### Question: A homeless man was walking down the street on a very cold night, when he entered an unlocked warehouse to get warm. There were some homeless families inside and he sat on the floor in a corner. He began to play with matches, lighting them for a bit of warmth, and then blowing them out. He carelessly threw them in a pile of refuse on the floor, which started smoldering. He got bored with that and looked around for something else to do. Meanwhile, the tiny fire grew, and the whole back of the building became ensnared in flames. Based on witness statements, the police arrested him for arson, which statutorily requires the "malicious and intentional" setting of fire to a structure. Will the prosecutor likely go forward with a prosecution for arson?\nA. Yes, because he was a trespasser and he disrespected the rights of the owner of the building.\nB. Yes, because he knew that he was creating an unreasonable risk of harm to others by playing with matches.\nC. No, because the owner of the structure made it too easy for outsiders to get inside the building, thus assuming the risk of the consequences.\nD. No, he did not intentionally set a fire or intend to burn the building, and he did not act maliciously.\n\n### Answer: ',
            },
        ],
        "ground_truth": "D",
        "prompt_length": 248,
    },
]


test_data_df = pd.DataFrame(test_data_values)

test_data_df.to_json(test_data, lines=True, orient="records")
