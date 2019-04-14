#Don't want to print all the info logs
import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

class keywordExpander():
    '''
    
    input: a list of projects with descriptions, dates, funding and outputs.
    
    -Expands keywords in a model (this could be clio or something else
    '''
    
    def __init__(self,projects,corpus_tokenised,model):
        '''
        
        Initialises the class with the projects and the w2v model we will use.
        
        '''
        
        self.projects = projects

        #This is the project df but with text description tokenised. We query it with the keywords 
        
        self.tokenised = corpus_tokenised
        self.w2v = model
        
        
    def keyword_expansion(self,seed_terms,thres):
        '''
        
        Expands a seed list of keywords. We input those as a dict where key corresponds to the type of input (solution or
        challenge, say) and the values are a list with the name of the entity (eg 'AI') and the seedlist to expand
        
        '''
        
        #Expand the keywords
        self.expanded_keywords = similarity_chaser(seed_terms,self.w2v,similarity=thres)
        
        return(self)
        

class keywordLabeller(keywordExpander):
    '''
    Classifies projects based on their keywords
    
    '''
    
    def __init__(self,missionKeywords):
        '''
        Initialise
        
        '''
        
        self.projects_labelled = missionKeywords.projects.copy()
        self.tokenised = missionKeywords.tokenised
        self.expanded_keywords = missionKeywords.expanded_keywords
    
    
    def filter_keywords(self,kws_to_drop):
        '''
        We can use it to drop irrelevant keywords. We 
        
        '''
                      
        #self.projects_labelled = self.projects_labelled[[x for x in self.projects_labelled.columns if x not in kws_to_drop]]
        
        self.expanded_keywords = [[x for x in kwset if x not in kws_to_drop] for kwset in self.expanded_keywords]
        
        
        return(self)
   
    
        
    
    def label_data(self,name,verbose=True):
        '''
        Queries the data with the keywords. It stores two attributes: the kw counts contain the counts by keywords 
        and challenge name; it also stores the projects labelled with an extra variable that
        counts the number of times that keywords in either set of keywords appears in the data.
        
        '''
        
        self.kw_counts = {}
        
        projects_labelled = self.projects_labelled
        tokenised = self.tokenised
        
        #We look for projects with keywords. Loop over name and extract the right index from the expanded keyword set.
        #This could also work as a dict.
        
            
        outputs = labeller(tokenised,self.expanded_keywords)
            
        self.kw_counts[name] = outputs[0]
            
        projects_labelled[name] = outputs[1]
            
        self.projects_labelled = projects_labelled
        
        return(self)
    

def similarity_chaser(seed_list,model,similarity,occurrences=1):
    '''
    Takes a seed term and expands it with synonyms (above a certain similarity threshold)
    
    '''
    
    #All synonyms of the terms in the seed_list above a certain threshold
    set_ws = flatten_list([[term[0] for term in model.most_similar(seed) if term[1]>similarity] for seed in seed_list])
    
    #return(set_ws)
    
    #This is the list of unique occurrences (what we want to return at the end)
    set_ws_list = list(set(set_ws))
    
    #For each term, if it appears multiple times, we expand
    for w in set_ws:
        if set_ws.count(w)>occurrences:
            
            #As before
            extra_words = [term[0] for term in model.wv.most_similar(w) if term[1]>similarity]
            
            set_ws_list + extra_words
            
    #return(list(set(set_ws_list)))
    #return(set_ws_list)
    return(set_ws)

    
def labeller(docs,keywords):
    '''
    Loops over a tokenised corpus and returns varios measures indicating the presence of keywords in it.
    
    This includes:
    
    * Whether at least one keyword appears
    * How many keywords appear
    * What keywords appear
    
    
    '''
    #Empty dict with keywords
    #kw_dict = {k:[] for k in keywords}
    
    #Loop through the corpus and create a vectorised df of the keywords
    
    out = pd.concat([pd.Series({x:(doc.count(x)) for x in keywords}) for doc in docs],axis=1).T
    
    
    #print(kw_dict)
    #Note that this also returns a sum of the keywords over the rows and a 
    return([out, out.sum(axis=1)])
    

#     #Intersection of tokens
#     if intersect==True:
    
#         out = [len(set(keywords) & set(document)) for document in corpus]
    
#     else:
#     #Otherwise it counts the total of tokens present in an abstract
        
#         out = [np.sum([x.count(k) for k in keywords]) for x in corpus]
    
    
    
    
    return(out)
    
    
def flatten_list(a_list):
    return([x for el in a_list for x in el])
