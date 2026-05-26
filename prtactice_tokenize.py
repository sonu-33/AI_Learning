

"""
from nltk import sent_tokenize
from nltk import word_tokenize
paragraph = '''
Hi, My name is Erin Neil.
I speak Mandarin Chienese.
I am from Texas.
I would love to visit Bhutan one day.
'''
print(paragraph)
print("#"*10)
print(sent_tokenize(paragraph))
print("#"*10)
print(word_tokenize(paragraph))

"""
#from nltk import word_tokenize

"""
from nltk.stem import PorterStemmer

stemming = PorterStemmer()

words = ["eating", "eats", "eaten", "history", "historian", "write", "writing"]

for word in words:
    print(word+"-->"+stemming.stem(word))
"""
"""
from nltk.stem import SnowballStemmer

stemming = SnowballStemmer("english")

words = ["eating", "eats", "eaten", "history", "historian", "write", "writing"]

for word in words:
    print(word + "-->"+ stemming.stem(word))

"""
# In the below code, we took a paragraph, convert that into tokens using word_tokenize
# Then, we will apply stopword, to remove all the stopwords from the token list
# Then we will apply lemmatization on token list
"""
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

paragraph = '''
Yesterday, I rushed into the office, answered three urgent calls, and promised that I would finish the report before lunch. 
While I was reviewing the numbers, my manager walks in, asks for an update, and reminds me that the client has been expecting 
quick results. By the end of the day, I will have sent the draft, corrected the errors, and prepared notes for tomorrow’s 
meeting. I have learned that plans change quickly, but I still try, adapt, and keep moving forward. If the feedback arrives 
early, I may revise the document again and present a stronger version by evening. 
On the edge of a small town, there was a bookstore that looked ordinary from the outside but felt strangely alive once you
stepped in. The front door rang with a soft bell that sounded different every time it opened. Dust floated
through the afternoon light like tiny actors performing in silence. The shelves were slightly uneven, and 
the floor creaked as if it had opinions about every visitor. In the back corner, a green armchair sat 
beside a lamp that never seemed to burn out. A sleepy orange cat usually occupied the chair, although no 
one remembered seeing it walk in. The owner wore mismatched socks and claimed it helped him organize his 
thoughts. He never arranged the books by genre, only by mood. On rainy days, people came in looking for 
mysteries and somehow left with poetry. On sunny mornings, the travel section became crowded with people 
who had no vacation plans at all. A cracked clock above the register was stuck at 4:17, but the owner 
insisted it was keeping track of a more interesting time zone. Sometimes customers found handwritten notes 
inside the books, offering advice, warnings, or recipes for soup. No one knew who wrote them, and no two 
notes sounded alike. A little girl once traded a marble for a paperback, and the owner accepted as if it 
were a rare coin. By evening, the shop smelled like paper, tea, and old wood warmed by the day. 
When the lights went out, the place felt less closed than quietly waiting for the next curious person to 
enter.
'''
word_list = word_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()

pre_process_list = []

for word in word_list:
    if word not in stopwords.words('english'):
        pre_process_list.append(lemmatizer.lemmatize(word, pos='v').lower())
    #pre_process_text = lemmatizer.lemmatize(word, pos='v').lower()
    #pre_process_list.append(pre_process_text)
    #print(word + " --> " +pre_process_text)

print(pre_process_list)
"""
# Below example is to find Pos tag from a paragraph

"""
import nltk
from nltk import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')

paragraph = '''
Yesterday, I rushed into the office, answered three urgent calls, and promised that I would finish the report before lunch. 
While I was reviewing the numbers, my manager walks in, asks for an update, and reminds me that the client has been expecting 
quick results. By the end of the day, I will have sent the draft, corrected the errors, and prepared notes for tomorrow’s 
meeting. I have learned that plans change quickly, but I still try, adapt, and keep moving forward. If the feedback arrives 
early, I may revise the document again and present a stronger version by evening. 
'''

tokenized_words = word_tokenize(paragraph)
#print (tokenized_words)

for word in tokenized_words:
    pos_tags = nltk.pos_tag(tokenized_words)

print(pos_tags)
"""
# Below code is usgage of nltk.pos_tag() 
# and nltk.#ne_chunk() combined to find the named
# entities in a paragraph
"""
import nltk
from nltk import word_tokenize

nltk.download('words')
nltk.download('maxent_ne_chunker_tab')

paragraph = '''
On March 15, 2026, Sarah Mitchell arrived in Chicago for a leadership workshop. She checked into her hotel near Millennium Park early that morning. Later that afternoon, Sarah met her colleague Daniel Reed for coffee to review their agenda. They planned to visit the client office on March 16 before returning home. By the end of the week, Sarah said April would be a better month for the next team gathering.
'''

tokenize_words = word_tokenize(paragraph)
pos_tag = nltk.pos_tag(tokenize_words)
print (pos_tag)

name_tag = nltk.ne_chunk(pos_tag)
print (name_tag)

"""