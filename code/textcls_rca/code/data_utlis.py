


def remove_punctaution(st):
    st = st.translate(str.maketrans(' ',' ', '!"#$%&\'()*+-./:;<=>?@[\\]^_`{|}~'))
    return st

def remove_digital(st):
    new_st = ''
    for char in st:
        if not char.isdigit():
            new_st = new_st + char
        else:
            new_st = new_st + ''
    return new_st

def remove_punc(st):
    punctuations = '''!()-[]{|};:'"\,<>./?@#$%^&*_`~=+'''
    new_st = ""
    for char in st:
        if char not in punctuations:
            new_st = new_st + char
        else:
            new_st = new_st + ' '

    return new_st