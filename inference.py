import pandas
from nltk.inference.resolution import Clause, BindingDict
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.sem import Valuation, Model
read_expr = Expression.fromstring
#######################################################
#  Initialise Knowledgebase.
#######################################################
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]


#========= Remember expression if no condriction =========
def remember(expr):
    answer=ResolutionProver().prove("", kb, verbose=False)
    if answer:
        kb.remove(expr)
        print("sorry it condradicts to what I know my friend")
    else:
        print('OK, I will remember that')

#========= Check for one of 3 answers ====================
def check(expr):
    answer=ResolutionProver().prove(expr, kb, verbose=False)
    if answer:
       print('Correct')
    else:               
       expr=read_expr('-' + str(expr))
       answer=ResolutionProver().prove(expr, kb, verbose=False)
       if answer:
           print("Incorrect")
       else:
           print("Sorry I don't know") 

def inference_remember(expersion):
    subject,predicate=expersion.split(' is ')
    expr=read_expr(predicate + '(' + subject + ')')
    if expr not in kb:
        kb.append(expr)
    remember(expr)

def inference_check(expersion):
    subject,predicate=expersion.split(' is ')
    expr=read_expr(predicate + '(' + subject + ')')
    check(expr);  


def inference_kb():
    for item in Clause(kb):
        print(item)