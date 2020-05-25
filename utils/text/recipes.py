from utils.files import get_files
from pathlib import Path
from typing import Union
from num2words import num2words

accents = {
        "è" : "{EH}",
        "ê" : "{EH}",
        "ë" : "{EH}",
        "é" : "{AX}",
        "à" : "a",
        "â" : "a",
        "æ" : "ae",
        "ç" : "ss",
        "î" : "i",
        "ï" : "{IH}",
        "ô" : "o",
        "œ" : "oe",
        "ù" : "u",
        "û" : "u",
        "ü" : "u",
        "ÿ" : "y"
    }
ke= [k for k in accents]
for k in ke:
    accents[k.upper()] = accents[k]


def french_preprocess(text):
    
    #process numbers
    no_numbers = []
    for word in text.split(" "):
        if word.isdigit():
            word = num2words(int(word), lang="fr")
        no_numbers.append(word)
    #process accents
    text = " ".join(no_numbers)
    cmu_accents = []
    for letter in text:
        if letter in accents:
            letter = accents[letter]
        cmu_accents.append(letter)
    return "".join(cmu_accents)



def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')

    assert len(csv_file) == 1

    text_dict = {}

    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            wav_file, text = split[0],split[-1]
            text = french_preprocess(text)
            text_dict[wav_file] = text

    return text_dict