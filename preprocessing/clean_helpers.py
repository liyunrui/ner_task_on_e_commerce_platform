import re # for removing emoji
import unicodedata # for deleting \xa0 Unicode representing spaces

def remove_emoji(string):
    '''
    obtained from https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python/41422178
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_name(s):
    '''
    It's an helper function for clean item_name.

    Can be modified for better
    Note:
        Cannot cleaa too much, especially including the notation related to key features.
        For example, do token contain hyphen? 
    parameters:
    --------------
    s: str   
    '''
    #----------------
    # prior knowledge
    #----------------
    # remove_emoji
    s = remove_emoji(s)
    # removing \xa0 Unicode representing spaces
    s = unicodedata.normalize("NFKD", s) # https://stackoverflow.com/questions/10993612/python-removing-xa0-from-string
    #----------------
    # new added
    #----------------    
    s = s.replace('!', '')
    s = s.replace('?', '')
    s = s.replace('*', '')
    s = s.replace('"',"")
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('/', '')
    s = s.replace('.', '')
    s = s.replace('.', '')
    s = s.replace('+', '')
    s = s.replace("'", '') 
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace(",", '')
    s = s.replace(';','')
    s = s.replace('{ ','')
    s = s.replace('} ','')
    s = s.replace('\n','')
    s = s.replace('\r','')
    return s

def clean_name_for_word_embedding(s):
    '''
    It's an helper function for clean item_name.

    Can be modified for better
    Note:
        Cannot cleaa too much, especially including the notation related to key features.
        For example, do token contain hyphen? 
    parameters:
    --------------
    s: str   
    '''
    #----------------
    # prior knowledge
    #----------------
    # remove_emoji
    s = remove_emoji(s)
    # removing \xa0 Unicode representing spaces
    s = unicodedata.normalize("NFKD", s) # https://stackoverflow.com/questions/10993612/python-removing-xa0-from-string
    #----------------
    # new added
    #----------------    
    s = s.replace('!', '')
    s = s.replace('?', '')
    s = s.replace('*', '')
    s = s.replace('"',"")
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('/', '')
    s = s.replace('.', '')
    s = s.replace('.', '')
    s = s.replace('+', '')
    s = s.replace("'", '') 
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace(",", '')
    s = s.replace(';','')
    s = s.replace('{ ','')
    s = s.replace('} ','')
    s = s.replace('\n','')
    s = s.replace('\r','')
    #----------------
    # extra adeed for get better word representation
    #----------------  
    s = s.replace('-','')
    s = s.replace('#','')
    return s