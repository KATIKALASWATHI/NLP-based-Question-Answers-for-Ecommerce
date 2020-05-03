
##########################lDA Model Building for Quastion and Answer####################################
import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

data2=pd.read_csv("file:///C:/Users/Swathi/Desktop/NLP-project/processed_data.csv")
data2.columns
stop=['yes','no']
data2['keywords']=data2['keywords'].apply(lambda x:" ".join(x for x in x.split() if x not in stop))

######LDA model for Question 
words=pd.Series(data1['keywords']).apply(lambda x: x.split()) ##taking tokenized words
dictionary=gensim.corpora.Dictionary(words) ##mapping words with their integer id's
count=0
for n, w in dictionary.iteritems():
    print(n, w)
    count +=1
    if count>100:
        break
#output:0 color
       1 cover
       2 fit
       3 nook
       4 glowlight....

doc_term_matrix=[dictionary.doc2bow(doc) for doc in words]##it shows how many times word appear in each document
bow_doc_x = doc_term_matrix[0]
for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], dictionary[bow_doc_x[i][0]], bow_doc_x[i][1]))

#output:Word 0 ("color") appears 1 time.
        Word 1 ("cover") appears 1 time.
        Word 2 ("fit") appears 1 time.
        Word 3 ("nook") appears 1 time.

####lda model
LDA=gensim.models.ldamodel.LdaModel
lda_model=LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5)
lda_model.print_topics()

vis=pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
pyLDAvis.show(vis)

###Output
[(0,
  '0.057*"work" + 0.025*"card" + 0.021*"drive" + 0.021*"laptop" + 0.020*"usb" + 0.017*"device" + 0.017*"tablet" + 0.017*"window" + 0.016*"computer" + 0.016*"port"'),
 (1,
  '0.020*"wifi" + 0.018*"tv" + 0.016*"pro" + 0.016*"warranty" + 0.016*"cable" + 0.015*"speaker" + 0.014*"box" + 0.013*"product" + 0.012*"power" + 0.012*"model"'),
 (2,
  '0.044*"fit" + 0.042*"case" + 0.033*"keyboard" + 0.020*"long" + 0.020*"power" + 0.012*"mount" + 0.011*"inch" + 0.011*"cord" + 0.010*"cover" + 0.009*"motherboard"'),
 (3,
  '0.037*"battery" + 0.026*"screen" + 0.015*"charge" + 0.014*"charger" + 0.013*"sound" + 0.013*"good" + 0.013*"time" + 0.011*"ram" + 0.011*"headphone" + 0.011*"turn"'),
 (4,
  '0.079*"camera" + 0.040*"work" + 0.035*"video" + 0.024*"lens" + 0.018*"iphone" + 0.018*"record" + 0.017*"compatible" + 0.013*"mm" + 0.012*"picture" + 0.012*"processor"')]





###LDA model for answer
words1=pd.Series(data1['Answer']).apply(lambda x: x.split())##tokenizing
dictionary1=corpora.Dictionary(words1)##mapping words with their integer id's
count=0
for n, w in dictionary1.iteritems():
    print(n, w)
    count +=1
    if count>100:
        break
#output:0 color
        1 fit
        2 nook
        3 sameshaped
        4 tablet.....

doc_term_matrix1=[dictionary1.doc2bow(doc) for doc in words]##shows no.of times word appears in each document
bow_doc_x = doc_term_matrix[1]
for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))
#output:Word 2 ("fit") appears 1 time.
        Word 3 ("nook") appears 1 time.
        Word 4 ("glowlight") appears 1 time.


LDA=gensim.models.ldamodel.LdaModel
lda_model1=LDA(corpus=doc_term_matrix1, id2word=dictionary1, num_topics=5)
lda_model1.print_topics()

vis1=pyLDAvis.gensim.prepare(lda_model1, doc_term_matrix1, dictionary1)
pyLDAvis.show(vis1)

####output
[(0,
  '0.035*"case" + 0.034*"drive" + 0.031*"work" + 0.027*"window" + 0.026*"fit" + 0.022*"laptop" + 0.020*"ipad" + 0.019*"monitor" + 0.015*"pro" + 0.015*"warranty"'),
 (1,
  '0.023*"video" + 0.021*"play" + 0.017*"time" + 0.016*"sound" + 0.013*"speaker" + 0.012*"audio" + 0.012*"record" + 0.011*"dvd" + 0.011*"good" + 0.010*"music"'),
 (2,
  '0.064*"work" + 0.035*"card" + 0.034*"tv" + 0.024*"keyboard" + 0.024*"tablet" + 0.018*"bluetooth" + 0.016*"wifi" + 0.015*"connect" + 0.012*"phone" + 0.012*"samsung"'),
 (3,
  '0.029*"battery" + 0.025*"usb" + 0.022*"power" + 0.020*"port" + 0.019*"gb" + 0.014*"screen" + 0.014*"cable" + 0.013*"long" + 0.012*"charge" + 0.011*"plug"'),
 (4,
  '0.078*"camera" + 0.038*"compatible" + 0.023*"lens" + 0.022*"work" + 0.015*"mount" + 0.013*"flash" + 0.013*"mm" + 0.012*"whats" + 0.011*"difference" + 0.011*"fit"')]
