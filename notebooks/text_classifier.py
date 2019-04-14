
#ML imports
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

import warnings

warnings.simplefilter('ignore',UserWarning)


from gensim import corpora, models
from string import punctuation
from string import digits
import re
import pandas as pd
import numpy as np

#Characters to drop
drop_characters = re.sub('-','',punctuation)+digits

#Stopwords
from nltk.corpus import stopwords

stop = stopwords.words('English')

#Stem functions
from nltk.stem import *
stemmer = PorterStemmer()


def clean_tokenise(string,drop_characters=drop_characters,stopwords=stop):
    '''
    Takes a string and cleans (makes lowercase and removes stopwords)
    
    '''
    

    #Lowercase
    str_low = string.lower()
    
    
    #Remove symbols and numbers
    str_letters = re.sub('[{drop}]'.format(drop=drop_characters),'',str_low)
    
    
    #Remove stopwords
    clean = [x for x in str_letters.split(' ') if (x not in stop) & (x!='')]
    
    return(clean)


class CleanTokenize():
    '''
    This class takes a list of strings and returns a tokenised, clean list of token lists ready
    to be processed with the LdaPipeline
    
    It has a clean method to remove symbols and stopwords
    
    It has a bigram method to detect collocated words
    
    It has a stem method to stem words
    
    '''
    
    def __init__(self,corpus):
        '''
        Takes a corpus (list where each element is a string)
        '''
        
        #Store
        self.corpus = corpus
        
    def clean(self,drop=drop_characters,stopwords=stop):
        '''
        Removes strings and stopwords, 
        
        '''
        
        cleaned = [clean_tokenise(doc,drop_characters=drop,stopwords=stop) for doc in self.corpus]
        
        self.tokenised = cleaned
        return(self)
    
    def stem(self):
        '''
        Optional: stems words
        
        '''
        #Stems each word in each tokenised sentence
        stemmed = [[stemmer.stem(word) for word in sentence] for sentence in self.tokenised]
    
        self.tokenised = stemmed
        return(self)
        
    
    def bigram(self,threshold=10):
        '''
        Optional Create bigrams.
        
        '''
        
        #Colocation detector trained on the data
        phrases = models.Phrases(self.tokenised,threshold=threshold)
        
        bigram = models.phrases.Phraser(phrases)
        
        self.tokenised = bigram[self.tokenised]
        
        return(self)
        
        
        
        

class LdaPipeline():
    '''
    This class processes lists of keywords.
    How does it work?
    -It is initialised with a list where every element is a collection of keywords
    -It has a method to filter keywords removing those that appear less than a set number of times
    
    -It has a method to process the filtered df into an object that gensim can work with
    -It has a method to train the LDA model with the right parameters
    -It has a method to predict the topics in a corpus
    
    '''
    
    def __init__(self,corpus):
        '''
        Takes the list of terms
        '''
        
        #Store the corpus
        self.tokenised = corpus
        
    def filter(self,minimum=5):
        '''
        Removes keywords that appear less than 5 times.
        
        '''
        
        #Load
        tokenised = self.tokenised
        
        #Count tokens
        token_counts = pd.Series([x for el in tokenised for x in el]).value_counts()
        
        #Tokens to keep
        keep = token_counts.index[token_counts>minimum]
        
        #Filter
        tokenised_filtered = [[x for x in el if x in keep] for el in tokenised]
        
        #Store
        self.tokenised = tokenised_filtered
        self.empty_groups = np.sum([len(x)==0 for x in tokenised_filtered])
        
        return(self)
    
    def clean(self):
        '''
        Remove symbols and numbers
        
        '''
        
        
        
    
        
    def process(self):
        '''
        This creates the bag of words we use in the gensim analysis
        
        '''
        #Load the list of keywords
        tokenised = self.tokenised
        
        #Create the dictionary
        dictionary = corpora.Dictionary(tokenised)
        
        #Create the Bag of words. This converts keywords into ids
        corpus = [dictionary.doc2bow(x) for x in tokenised]
        
        self.corpus = corpus
        self.dictionary = dictionary
        return(self)
        
    def tfidf(self):
        '''
        This is optional: We extract the term-frequency inverse document frequency of the words in
        the corpus. The idea is to identify those keywords that are more salient in a document by normalising over
        their frequency in the whole corpus
        
        '''
        #Load the corpus
        corpus = self.corpus
        
        #Fit a TFIDF model on the data
        tfidf = models.TfidfModel(corpus)
        
        #Transform the corpus and save it
        self.corpus = tfidf[corpus]
        
        return(self)
    
    def fit_lda(self,num_topics=20,passes=5,iterations=75,random_state=1803):
        '''
        
        This fits the LDA model taking a set of keyword arguments.
        #Number of passes, iterations and random state for reproducibility. We will have to consider
        reproducibility eventually.
        
        '''
        
        #Load the corpus
        corpus = self.corpus
        
        #Train the LDA model with the parameters we supplied
        lda = models.LdaModel(corpus,id2word=self.dictionary,
                              num_topics=num_topics,passes=passes,iterations=iterations,random_state=random_state)
        
        #Save the outputs
        self.lda_model = lda
        self.lda_topics = lda.show_topics(num_topics=num_topics)
        

        return(self)
    
    def predict_topics(self):
        '''
        This predicts the topic mix for every observation in the corpus
        
        '''
        #Load the attributes we will be working with
        lda = self.lda_model
        corpus = self.corpus
        
        #Now we create a df
        predicted = lda[corpus]
        
        #Convert this into a dataframe
        predicted_df = pd.concat([pd.DataFrame({x[0]:x[1] for x in topics},
                                              index=[num]) for num,topics in enumerate(predicted)]).fillna(0)
        
        self.predicted_df = predicted_df
        
        return(self)
    


# CLasses





#One class for text classification based on text inputs

class TextClassification():
    '''
    This class takes a corpus (could be a list of strings or a tokenised corpus) and a target (could be multiclass or single class).
    
    When it is initialised it vectorises the list of tokens using sklearn's count vectoriser.
    
    It has a grid search method that takes a list of models and parameters and trains the model.
    
    It returns the output of grid search for diagnosis
    
    '''
    
    def __init__(self,corpus,target):
        '''
        
        Initialise. The class will recognise if we are feeding it a list of strings or a list of
        tokenised documents and vectorise accordingly. 
        
        It will also recognise is this a multiclass or one class problem based on the dimensions of the target array
        
        Later on, it will use control flow to modify model parameters depending on the type of data we have
        
        '''
        
        #Is this a multiclass classification problem or a single class classification problem?
        if target.shape[1]>1:
            self.mode = 'multiclass'
            
        else:
            self.mode = 'single_class'
    
    
        #Store the target
        self.Y = target
    
        #Did we feed the model a bunch of strings or a list of tokenised docs? If the latter, we clean and tokenise.
        
        if type(corpus[0])==str:
            corpus = CleanTokenize(corpus).clean().bigram().tokenised
            
        #Turn every list of tokens into a string for count vectorising
        corpus_string =  [' '.join(words) for words in corpus]
        
        
        #And then we count vectorise in a hacky way.
        count_vect = CountVectorizer(stop_words='english',min_df=5).fit(corpus_string)
        
        #Store the features
        self.X = count_vect.transform(corpus_string)
        
        #Store the count vectoriser (we will use it later on for prediction on new data)
        self.count_vect = count_vect
        
    def grid_search(self,models):
        '''
        The grid search method takes a list with models and their parameters and it does grid search crossvalidation.
        
        '''
        
        #Load inputs and targets into the model
        Y = self.Y
        X = self.X
        
        if self.mode=='multiclass':
            '''
            If the model is multiclass then we need to add some prefixes to the model paramas
            
            '''
        
            for mod in models:
                #Make ovr
                mod[0] = OneVsRestClassifier(mod[0])
                
                #Add the estimator prefix
                mod[1] = {'estimator__'+k:v for k,v in mod[1].items()}
                
        
        #Container with results
        results = []

        #For each model, run the analysis.
        for num,mod in enumerate(models):
            print(num)

            #Run the classifier
            clf = GridSearchCV(mod[0],mod[1])

            #Fit
            clf.fit(X,Y)

            #Append results
            results.append(clf)
        
        self.results = results
        return(self)

    
#Class to visualise the outputs of multilabel models.

#I call it OrangeBrick after YellowBrick, the package for ML output visualisation 
#(which currently doesn't support multilabel classification)


class OrangeBrick():
    '''
    This class takes a df with the true classes for a multilabel classification exercise and produces some charts visualising findings.
    
    The methods include:
    
        .confusion_stack: creates a stacked barchart with the confusion matrices stacked by category, sorting classes by performance
        .prec_rec: creates a barchart showing each class precision and recall;
        #Tobe done: Consider mixes between classes?
    
    '''
    
    def __init__(self,true_labels,predicted_labels,var_names):
        '''
        Initialise with a true labels, predicted labels and the variable names
        '''
         
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.var_names = var_names
    
    def make_metrics(self):
        '''
        Estimates performance metrics (for now just confusion charts by class and precision/recall scores for the 0.5 
        decision rule.
        
        '''
        #NB in a confusion matrix in SKlearn the X axis indicates the predicted class and the Y axis indicates the ground truth.
        #This means that:
            #cf[0,0]-> TN
            #cf[1,1]-> TP
            #cf[0,1]-> FN (prediction is false, groundtruth is true)
            #cf[1,0]-> FP (prediction is true, ground truth is false)



        #Predictions and true labels
        true_labels = self.true_labels
        pred_labels = self.predicted_labels

        #Variable names
        var_names = self.var_names

        #Store confusion matrices
        score_store = []


        for num in np.arange(len(var_names)):

            #This is the confusion matrix
            cf = confusion_matrix(pred_labels[:,num],true_labels[:,num])

            #This is a melted confusion matrix
            melt_cf = pd.melt(pd.DataFrame(cf).reset_index(drop=False),id_vars='index')['value']
            melt_cf.index = ['true_negative','false_positive','false_negative','true_positive']
            melt_cf.name = var_names[num]
            
            #Order variables to separate failed vs correct predictions
            melt_cf = melt_cf.loc[['true_positive','true_negative','false_positive','false_negative']]

            #We are also interested in precision and recall
            prec = cf[1,1]/(cf[1,1]+cf[1,0])
            rec = cf[1,1]/(cf[1,1]+cf[0,1])

            prec_rec = pd.Series([prec,rec],index=['precision','recall'])
            prec_rec.name = var_names[num]
            score_store.append([melt_cf,prec_rec])
    
        self.score_store = score_store
        
        return(self)
    
    def confusion_chart(self,ax):
        '''
        Plot the confusion charts
        
        
        '''
        
        #Visualise confusion matrix outputs
        cf_df = pd.concat([x[0] for x in self.score_store],1)

        #This ranks categories by the error rates
        failure_rate = cf_df.apply(lambda x: x/x.sum(),axis=0).loc[['false' in x for x in cf_df.index]].sum().sort_values(
            ascending=False).index

        
        #Plot and add labels
        cf_df.T.loc[failure_rate,:].plot.bar(stacked=True,ax=ax,width=0.8,cmap='Accent')

        ax.legend(bbox_to_anchor=(1.01,1))
        #ax.set_title('Stacked confusion matrix for disease areas',size=16)
    
    
    def prec_rec_chart(self,ax):
        '''
        
        Plot a precision-recall chart
        
        '''
    

        #Again, we sort them here to assess model performance in different disease areas
        prec_rec = pd.concat([x[1] for x in self.score_store],1).T.sort_values('precision')
        prec_rec.plot.bar(ax=ax)

        #Add legend and title
        ax.legend(bbox_to_anchor=(1.01,1))
        #ax.set_title('Precision and Recall by disease area',size=16)