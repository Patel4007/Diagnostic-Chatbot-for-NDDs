# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import math
import pickle
import random
import smtplib
import urllib.parse
import webbrowser
import numpy as np
from requests_html import HTMLSession

import spacy
import nltk
import sys
import pyrebase
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
from heapq import nlargest
import MySQLdb
import re
from newspaper import Article
from nltk.tokenize.punkt import PunktSentenceTokenizer
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.events import FollowupAction
import string
from rasa_sdk.types import DomainDict
from mysql.connector import Error
from typing import Any, Text, Dict, List, Union
from rasa.core.tracker_store import DialogueStateTracker
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pytorch_lightning as pl
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer)

MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

unique_key = sys.argv[1]

pickle.dump(unique_key, open("passwd.data", "wb"))
passwd = pickle.load(open("passwd.data", "rb"))


# Function to establish a connection to a MySQL database
def DataConnect(host_name, root, password, db_name):
    mydb = None

    try:
        mydb = MySQLdb.connect(host_name, root, password, db_name)

    except Error as e:
        print(e)

    return mydb


mydb = DataConnect("localhost", "root", "Patel_4007", "rasa_input")


# Function to insert user data into the 'users' table
def Users(mydb, userName, email, password, Relation):
    mycursor = mydb.cursor()

    # sql = "CREATE TABLE Input_rasa (personName VARCHAR(255), relation VARCHAR(255), input VARCHAR(255));"

    sql = 'insert into users(userName, email password, relation) values ("{0}","{1}","{2}","{3}");'.format(
        userName, email, password, Relation)

    mycursor.execute(sql)

    mydb.commit()

    print(mycursor.rowcount, "record inserted")

    mydb.close()

# Function to insert patient data into the 'patient_data' table
def PatientData(mydb, email, PersonName, summary, motor_skills, ehcd, compulsive):
    mycursor = mydb.cursor()

    # sql = "CREATE TABLE Input_rasa (personName VARCHAR(255), relation VARCHAR(255), input VARCHAR(255));"

    sql = 'insert into patient_data(email, personName, summary, motor_skills, eye_hand_coord, compulsive) values ("{0}","{1}","{2}","{3}","{4}", "{5}");'. \
        format(email, PersonName, summary, motor_skills, ehcd, compulsive)

    mycursor.execute(sql)

    mydb.commit()

    print(mycursor.rowcount, "record inserted")

    mydb.close()


modelPath = "C:\\Users\\JAY PATEL\\Desktop"



# Firebase configuration details 
config = {

  "apiKey": {{apikey}},
  "authDomain": {{authDomain}},
  "projectId": {{projectId}},
  "storageBucket": {{storageBucket}},
  "messagingSenderId": {{senderId}},
  "appId": {{appId}},
  "measurementId": {{mId}},
  "databaseURL": {{dbURL}}

}

firebase = pyrebase.initialize_app(config)

# Function to clean input text by removing punctuation, converting to lowercase, and filtering stopwords
def clean_string(text):

    text = ' '.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in nltk.word_tokenize(text) if word not in stopwords.words('english')])

    return text

# Function to calculate cosine similarity between two vectors
def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


# Function to execute a query on the database and fetch results
def DataQuery(mydb, query):

    cursor = mydb.cursor()

    cursor.execute(query)

    para1 = cursor.fetchall()

    cursor.close()

    return para1


# PyTorch Lightning model configuration

class DiagnosticModel(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
 
        return output.loss, output.logits

    def train_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return loss
     
    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0009)

    

diagnosticModel = DiagnosticModel


# Action to greet the user and extract their name
class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        name = tracker.get_slot("first")

        nlp = spacy.load("en_core_web_sm")

        doc = nlp(str(tracker.latest_message['text']))

        for word in doc.ents:
            if len(name) > 2 and word.label_ == 'PERSON':

                n = word.text

                db = firebase.database()

                message = "nice to meet you {}".format(n)

                dispatcher.utter_message(text=message)
                dispatcher.utter_message(text="How are you related to the autistic individual, for example: are you a mother, father or brother")

                data = {"name": n}

                db.child("users").child("ownkey").set(data)

                return [SlotSet('first', n)]

# Action to extract and validate family relation
class ActionRelation(Action):

    def name(self) -> Text:
        return "action_relation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        w = tracker.get_slot("first")
        m = str(tracker.latest_message['text'])
        family_list = ['mother', 'father', 'brother', 'sister', 'grandfather', 'grandmother',
                       'teacher', 'guardian', 'cousin', 'friend', 'nephew', 'neice', 'uncle', 'aunt']

        s = re.findall(r"\w+", m)

        e = ""

        for w in s:
            for a in family_list:
                if w.lower() == a:
                    e = w.lower()
                    break

        dispatcher.utter_message(text="what is the name of your child who has autism issues ?")

        return [SlotSet('first',w),SlotSet('relation',e)]


# Action to extract the patient's name
class ActionPatient(Action):

    def name(self) -> Text:
        return "action_patient_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        e = tracker.get_slot('first')
        m = tracker.get_slot('relation')

        name = tracker.get_slot("patient")

        nlp = spacy.load("en_core_web_sm")

        doc = nlp(str(tracker.latest_message['text']))

        for word in doc.ents:
            if len(name) > 2 and word.label_ == 'PERSON':

                n = word.text

                dispatcher.utter_message(response="utter_mail")


                return [SlotSet('patient', n), SlotSet('first', e), SlotSet('relation',m)]

# Action to handle OTP verification
class ActionOTP(Action):

    def name(self) -> Text:
        return "action_otpwrd"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        b = tracker.get_slot("email")
        w = tracker.get_slot("otpwd")

        m = str(tracker.latest_message['text'])

        query = "select password from users where email = '{0}';".format(b)

        wp = DataQuery(mydb, query)

        output = ''.join([''.join(map(str, tup)) for tup in wp])

        if (m == w and len(output) > 0):

            dispatcher.utter_message(text="your account password is " + output)
            dispatcher.utter_message(text="type 'login' to sign in your account")


        return [SlotSet("email", b)]


# Action to handle password validation and account creation
class ActionPasswrd(Action):

    def name(self) -> Text:
        return "action_passwrd"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        e = tracker.get_slot("email")
        patient_name = tracker.get_slot("patient")
        name = tracker.get_slot("first")
        relations = tracker.get_slot("relation")

        f = str(tracker.latest_message['text'])

        query = "select email from users where email = '{0}';".format(e)

        wp = DataQuery(mydb, query)

        queries = "select password from users where password = '{0}';".format(f)

        p = DataQuery(mydb, queries)

        output = ''.join([''.join(map(str, tup)) for tup in wp])

        outputs = ''.join([''.join(map(str, tup)) for tup in p])

        characters = "0123456789abcdefghijklmnopkrstuvwxyz"


        if (len(output) > 0 and len(outputs) > 0):

            dispatcher.utter_message("Welcome")

        elif(len(output) > 0 and len(outputs) == 0):

            dispatcher.utter_message("invalid password")

            s = smtplib.SMTP('smtp.gmail.com', 587)

            s.starttls()

            s.login("patel0047700@gmail.com", "P/t&l_7004")

            otp = ""

            length = len(characters)

            for i in range(5):
                otp += characters[math.floor(random.random() * length)]

            message = "\nHello, here is your otp: {}".format(otp)

            s.sendmail("patel41100@gmail.com", output, message)

            s.quit()

            dispatcher.utter_message(text="An email has been sent to you. enter the otp received to access the password")

            return [SlotSet("otpwd", otp), SlotSet("email", e)]


        elif (len(outputs) == 0 and len(output) == 0):

            Users(mydb, name, e, f, relations)
            PatientData(mydb, e, patient_name, "", "", "", "")

            dispatcher.utter_message("account created successfully")


        else:
            dispatcher.utter_message("wrong input")
            dispatcher.utter_message("type 'sign up' to retry login")



        return [SlotSet('email', e)]


# Action to handle email verification and account setup
class ActionMail(Action):

    def name(self) -> Text:
        return "action_mail"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        f = tracker.get_slot('first')
        y = tracker.get_slot('relation')
        d = tracker.get_slot('patient')

        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        digits = "0123456789"

        name = tracker.get_slot("email")

        m = str(tracker.latest_message['text'])

        lsm = re.findall(regex, m)

        v = ''.join(lsm)

        if len(name) > 0 and len(lsm) == 1:

            query = "select email from users where email = '{0}';".format(v)

            wp = DataQuery(mydb, query)

            output = ''.join([''.join(map(str, tup)) for tup in wp])

            if len(output) > 0:

                dispatcher.utter_message(text="Enter the password")

                return [SlotSet("email", output)]

            else:

                dispatcher.utter_message(text="email id doesn't exist in our database")

                dispatcher.utter_message(text="Enter a password to create an account")

                return [SlotSet("email", v), SlotSet('patient', d), SlotSet('first', f), SlotSet('relation', y)]
        else:

            dispatcher.utter_message(text="enter a valid email address")
            dispatcher.utter_message(response="utter_mail")


# This action is executed when a fallback response is triggered.
class ActionCustomFallback(Action):

    def name(self) -> Text:
        return "action_custom_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: DialogueStateTracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        users = tracker.get_slot('email')

        m = tracker.latest_message['text']
        u = tracker.get_latest_entity_values("check_similarity")
        v = list(u)
        entities_to_process = []

        entities_to_process.extend(v)

        return[SlotSet('email', users), FollowupAction("action_similarity")]


# Action to handle attention-related tasks, update database and process user inputs.
class ActionAttention(Action):

    def name(self) -> Text:
        return "action_attention"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        users = tracker.get_slot("email")

        m = str(tracker.latest_message['entities'])

        if m == "similarity":

            cursor = mydb.cursor()

            while cursor.nextset() is not None: pass

            query = "update patient_data set motor_skills = concat(ifnull(motor_skills,''),'fail') where email = '{0}';".format(users)
            cursor.execute(query)
            mydb.commit()
            print("row updated")

        elif m == "compulsive":

            cursor = mydb.cursor()

            while cursor.nextset() is not None: pass

            query = "update patient_data set compulsive = concat(ifnull(compulsive,''),'fail') where email = '{0}';".format(users)

            cursor.execute(query)
            mydb.commit()
            print("row updated")

        else:
            cursor = mydb.cursor()

            while cursor.nextset() is not None: pass

            query = "update patient_data set eye_hand_coord = concat(ifnull(eye_hand_coord,''),'fail') where email = '{0}';".format(users)

            cursor.execute(query)
            mydb.commit()
            print("row updated")

        e = str(tracker.latest_message['text']) + "."

        nlp = spacy.load('en_core_web_sm')

        v = nlp(e)

        doc = nlp(' '.join([str(t) for t in v if not t.is_stop]))

        questionone = generate_response(doc)

        sims = []
        for token in questionone:
            sims.append(doc.similarity(nlp(' '.join([str(t) for t in nlp(token) if not t.is_stop]))))

        id_max = np.argmax(sims)

        dispatcher.utter_message(text=questionone[id_max],
                                     buttons=[
                                         {"title": "yes", "payload": "/affirm"},
                                         {"title": "no", "payload": "/deny"}
                                     ])

        return []

    def validate_attention(
            self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> Union[Dict[str, bool], Dict[str, None]]:

        if tracker.get_intent_of_latest_message() == "affirm":
            return {"attention": True}
        if tracker.get_intent_of_latest_message() == "deny":
            return {"attention": False}
        dispatcher.utter_message(text="I didn't get that")
        return {"attention": None}


class LoginAction(Action):

    def name(self) -> Text:
        return "action_login"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Hi, I am Rasa. Login or Create Account",
                                 buttons=[
                                     {"title": "login", "payload": "/login"},
                                     {"title": "create account", "payload": "/create_account"}
                                 ])

        return []


# Action to handle similarity-related logic and update the database fields.
class Similarity(Action):


    def name(self) -> Text:
        return "action_similarity"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user = tracker.get_slot('email')


        nlp = spacy.load("en_core_web_sm")

        many_words = words.words() + wordnet.words()

        validate_events = []


        tokenizer = TreebankWordTokenizer()

        user_text = tracker.latest_message['text']

        tokenized_list = tokenizer.tokenize(user_text)

        for x in tokenized_list:
            if x not in many_words:
                validate_events.append(x)

        if len(validate_events) > 0:
            dispatcher.utter_message(text="Sorry, I didn't understand Try Again")


        e = str(user_text) + "."
        m = nlp(e)

        questions = generate_response(m)


        if len(user_text) > 2:

            sims = []
            for token in nlp.pipe(questions):
                sims.append(m.similarity(token))

            id_max = np.argmax(sims)

            dispatcher.utter_message(text="Thanks for the input. You can access the summary by typing 'I want summary'")

        cursor = mydb.cursor()

        while cursor.nextset() is not None: pass
        querys = "update patient_data set summary = concat(summary, '{0}') where email = '{1}';".format(m, user)
        cursor.execute(querys)
        mydb.commit()
        print("row updated")


        return [SlotSet('email', user), FollowupAction("action_summary")]


# Function to generate a response using a pre-trained model

def generate_response(input):
    source_encoding = tokenizer(

        input["input"],
        input["context"],
        max_length = 150,
        padding = "max_length",
        return_attention_mask = True,
        add_special_token = True,
        return_tensors = "pt"     
    )

    generated_ids = diagnosticModel.model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask = source_encoding["attention_mask"],
        num_beams = 1,
        max_length = 70,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True
    )

    predictions = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                  for generated_id in generated_ids]

    return "".join(predictions)


# Custom Rasa action to respond based on user input
class ActionResponse(Action):

    def name(self) -> Text:
        return "action_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:

        use_email = tracker.get_slot("email")

        max_question = generate_response(tracker.latest_message['text'])


        if tracker.get_slot("attention") == True:
            dispatcher.utter_message(text="Do you find any type of different behaviour displayed by your kid ?")

        elif tracker.get_slot("attention") == False:

            dispatcher.utter_message(text=max_question,
                                     buttons=[
                                         {"title": "yes", "payload": "/affirm"},
                                         {"title": "no", "payload": "/deny"},
                                     ])
        else:
            dispatcher.utter_message(text = "No proper information")


        return []

    def validate_response(
            self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> Union[Dict[str, bool], Dict[str, None]]:

        if tracker.get_intent_of_latest_message() == "affirm":
            return {"response": True}
        if tracker.get_intent_of_latest_message() == "deny":
            return {"response": False}
        dispatcher.utter_message(text="I didn't get that")
        return {"response": None}


# Action to handle the final output based on the 'response' slot
class ActionFinal(Action):

    def name(self) -> Text:
        return "action_final_output"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        if tracker.get_slot("response") == True:
            dispatcher.utter_message(text="Tell me more about your kid's behaviour and activities")

        elif tracker.get_slot("response") == False:
            preds = generate_response(tracker.latest_message['text'])
            dispatcher.utter_message(text=preds)
        else:
            dispatcher.utter_message(text="No proper information")

        return []


# Action to summarize patient data and provide relevant information
class ActionSummary(Action):

    def name(self) -> Text:
        return "action_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        uses = tracker.get_slot('email')


        stopwords = list(STOP_WORDS)


        query = "select summary from patient_data where email = {0};".format(uses)

        wp = DataQuery(mydb, query)

        output = ''.join([''.join(map(str, tup)) for tup in wp])

        nlp = spacy.load('en_core_web_sm')

        preds = generate_response(tracker.latest_message.text)


        doc = nlp(str(output))

        word_dictionary = {}

        for word in doc:
            if word.text.lower() not in stopwords:
                if word.text.lower() not in punctuation:
                    if word.text not in word_dictionary.keys():
                        word_dictionary[word.text] = 1
                    else:
                        word_dictionary[word.text] += 1

        max_frequency = max(word_dictionary.values())

        for word in word_dictionary.keys():
            word_dictionary[word] = word_dictionary[word]/max_frequency

        sentence_tokens = [sent for sent in doc.sents]

        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_dictionary.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_dictionary[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_dictionary[word.text.lower()]

        select_length = int(len(sentence_tokens)*0.5)
        summaries = nlargest(select_length, sentence_scores, key = sentence_scores.get)
        final_summary = [word.text for word in summaries] + preds
        summary = ' '.join(final_summary)
        e = "NDD treatment " + summary
        medical_reply = response(e)
        y = "home remedies " + summary + " autism"
        temperory_solution = response(y)

        dispatcher.utter_message(text="summary output: " + summary + "\n\n" + medical_reply + "\n\n" + temperory_solution)



        return []

def index_sort(list_var):

    length = len(list_var)
    list_index = list(range(0,length))

    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:

                list_index[i],list_index[j] = list_index[j],list_index[i]

    return list_index

# Function to curate data from the internet and generate a response  
def response(user_input):

    user_input = '+'.join(user_input.split())

    article = Article(webbrowser.get('google'), user_input)
    article.download()
    article.parse()
    article.nlp()
    corpus = article.text
    text = corpus
    text = text.translate(str.maketrans('', '', string.punctuation))
    sent_tokenizer = PunktSentenceTokenizer(text)
    sentence_list = sent_tokenizer.tokenize(text)  # nltk.sent_tokenize(text)

    user_input = user_input.lower()
    response = ''

    count_matrix = CountVectorizer(stop_words=["are", "all", "in", "and"]).fit_transform(
        sentence_list)  # (number of articles, number of unique words)
    array = count_matrix.toarray()
    similarity_matrix = cosine_similarity(count_matrix[-1], count_matrix)
    sparse_array = similarity_matrix.flatten()
    index = index_sort(sparse_array)
    index = index[1:]
    response_flag = 0
    j = 0

    for i in range(len(index)):
        if sparse_array[index[i]] > 0.0:
            sentence = ' '.join([str(element) for element in sentence_list])
            response = response + sentence
            response_flag = 1
            j = j + 1

        if response_flag == 0:
            response = response + ' ' + 'I did not understand'

        sentence = ' '.join([str(element) for element in sentence_list])

        response = ' '.join(response.split()[:32])

        b = urllib.parse.quote_plus(response)

        article = ''

        session = HTMLSession()

        r = session.get("https://www.google.com/search?q=" + b)

        links = list(r.html.absolute_links)

        google_domains = (
            'https://www.google.',
            'https://google.',
            'https://webcache.googleusercontent.',
            'http://webcache.googleusercontent.',
            'https://policies.google.',
            'https://support.google.',
            'https://maps.google.')

        for url in links[:]:
            if url.startswith(google_domains):
                links.remove(url)

        m = links[0]
        m = str(m)
        a = Article(m)
        a.download()
        a.parse()
        a.nlp()

        w = a.summary
        q = w + "\n\n" + "more information is available on this website: " + "\n" + m

        return q

    print('\n')