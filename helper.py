from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
from langchain import OpenAI
import openai
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA 
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.memory import PostgresChatMessageHistory
import requests
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from dataCreation import get_docs
load_dotenv(override=True)

# initialize LLM
def initialize_llm():
    global llm
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY") , temperature=0.6)
    # llm = OpenAI()




# embeddings
def initialize_embeddings():
    try :
        global embeddings
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name="sentence-transformers/all-MiniLM-l6-v2"
        )
    except Exception as e :
        print("probelm accured while creating embeddings \n{}".format(e))


# vector database
def load_vectorDB(collection_name):
    global vector_db
    vector_db = Milvus(
    embeddings,
    collection_name=collection_name,
    connection_args={
    "uri":os.getenv("ZILLIZ_CLOUD_URI") ,
    "user":os.getenv("ZILLIZ_CLOUD_USERNAME") ,
    "password":os.getenv("ZILLIZ_CLOUD_PASSWORD") ,
    # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
    "secure": True,
    },
)
    

def get_template(query,user_name ,user_email,date):
    #  get tags
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    functionCode = model(
        [
            SystemMessage(content="""you are a function controller. you are working at customer care center. based on the provided query below choose appropriate function to call. only output the function code provided below. 
                          function: "need human help"
                          code: "$$helpH$$" 
                          function: "extremely frustrated" 
                          code: "$$contactSoon$$" 
                          function: "normal conversation"
                          code: "$$continue$$" 
                          query : """),
            HumanMessage(content=query),
        ])
    functionCode = str(functionCode).split("$$")[1]

    print("functionCode ---->" , functionCode)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    userid = f"{user_name}_{timestamp}"

    if (functionCode == "helpH"):
        print("requested human help ...")
        url = 'https://brainxchatbot.vercel.app/api/contact'
        headers = {
        "userID": userid,
        "condition": "assistance",
        "userName": user_name,
        "userEmail": user_email,
        "lastDate": "25/7",
      }
        response = requests.post(url, headers=headers, data={})
        print("response ====> ",response)

        return  {
            "template" : "" , 
            "functionCode" : "helpH",
            "message": "Please wait for some time. Our customer representative would contact you shortly. Meanwhile if you want you can contact us on reach@brainx.com."}
    else : 
        if (functionCode == "contactSoon"):
            print("angry user ...")
            return  {
                "template" : "" , 
                "functionCode" : "contactSoon",
                "message": "We are extremely sorry for causing the inconvience. Our customer representative would contact you shortly. If you want you can write us on react@brainx.com also."}
        else:
            if (functionCode == "continue"):
                simtag = vector_db.similarity_search(query=query , k=3)
                simmcontext = simtag[0].page_content + " --- " + simtag[1].page_content + " --- " + simtag[2].page_content    
                print("sim context " , simmcontext )
                return {
                    "template" : """ you are a salse person. try to give human like response according to user query and given context.
                Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>)   to answer the question. only provide accurate answers, and if you don't know the answer ask user to mail at contact@brainx.com  :
                ------
                <ctx> """ + simmcontext + """
                {context}
                </ctx>
                ------
                <hs>
                {history}
                </hs>
                ------
                {question}
                Answer:
                """ , 
                "functionCode" : "continue",
                "message": "ok"}
            else : 
                print("problem with function controller")
                return {
                "template" : "" , 
                "functionCode" : "error",
                "message": "something went wrong, contact us at contact@brainx.com"}
            

def ask_ai(companyName , query,user_name ,user_email,date):
    # load vector db
    load_vectorDB(companyName)
    templateObject = get_template(query,user_name ,user_email,date)
    print("\n\n\nprompt ----------- ",templateObject)
    if templateObject["functionCode"] == "continue":
        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=templateObject['template'],
        )
        # history
        history = PostgresChatMessageHistory(
            connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
            session_id=companyName,)
        user_memory = ConversationBufferWindowMemory(k=3 , memory_key="history" , chat_memory= history, input_key="question")
        posgres_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vector_db.as_retriever(),
            # verbose=True,
            chain_type_kwargs={
                # "verbose": True,
                "prompt": prompt,
                "memory": user_memory
            }
        )
        # print("prompt -------------------------- " , vector_db.similarity_search(query=query , k=10))
        res = posgres_qa(query)
        return res["result"]
    else : 
        return templateObject["message"]