import datetime
import numpy as np


def get_latest_file(date_list,date_format='%d-%m-%Y'):
    '''
    This function takes a list of date strings and returns the most recent one
    
    Args:
        date_list: a list of strings with the format date_filename
        date_format: the format for the date, defaults to %d-%m-%Y
    
    Returns:
        The element with the latest date
    
    '''
    
    #This gets the maximum date in the gtr directory
    dates = [datetime.datetime.strptime('-'.join(x.split('_')[:3]),date_format) for x in date_list]
    
    #Return the most recent file
    most_recent = sorted([(x,y) for x,y in zip(date_list,dates)],key=lambda x:x[1])[0][0]
    
    return(most_recent)
                                        
    
# Put functions and classes here

def flatten_list(a_list):
    return([x for el in a_list for x in el])


def random_check(corpus,num,length):
    '''
    Prints num random examples form corpus
    
    '''
    
    selected = np.random.randint(0,len(corpus),num)
    
    texts  = [text for num,text in enumerate(corpus) if num in selected]
    
    for t in texts:
        print(t[:length])
        print('====')