from langchain_neo4j import Neo4jGraph
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory
from uuid import uuid4


load_dotenv()

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# response = llm.invoke("What is Neo4j?")

# print(response)

# ---------

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

# response = llm.invoke(template.format(fruit="apple"))

# print(response)

# --------- OR ---------

# llm_chain = template | llm | StrOutputParser()

# response = llm_chain.invoke({"fruit": "apple"})

# print(response)


# --------------------------------------------------------
# Chat Models vs Language Models
# --------------------------------------------------------

# Until now, you have been using a language model to communicate with the LLM.
# A language model predicts the next word in a sequence of words. Chat models are designed to have conversations - they accept a list of messages and return a conversational response.
# Chat models typically support different types of messages:
#     System - System messages instruct the LLM on how to act on human messages
#     Human - Human messages are messages sent from the user
#     AI - Responses from the AI are called AI Responses

# instructions = SystemMessage(content="""
# You are a surfer dude, having a conversation about the surf conditions on the beach.
# Respond using surfer slang.
# """)
# question = HumanMessage(content="What is the weather like?")

# response = llm.invoke([
#     instructions,
#     question
# ])

# print(response.content)

# --------- OR ---------

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
#         ),
#         (
#             "human",
#             "{question}"
#         ),
#     ]
# )

# chat_chain = prompt | llm | StrOutputParser()

# response = chat_chain.invoke({"question": "What is the weather like?"})

# print(response)

# ---------------- additonal content ----------------

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
#         ),
#         ( "system", "{context}" ),
#         ( "human", "{question}" ),
#     ]
# )

# chat_chain = prompt | llm | StrOutputParser()

# current_weather = """
#     {
#         "surf": [
#             {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
#             {"beach": "Polzeath", "conditions": "Flat and calm"},
#             {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
#         ]
#     }"""

# response = chat_chain.invoke(
#     {
#         "context": current_weather,
#         "question": "What is the weather like on Watergate Bay?",
#     }
# )

# print(response)

# --------------------------------------------------------
# Giving the model memory
# --------------------------------------------------------

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
#         ),
#         ("system", "{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ]
# )

# memory = ChatMessageHistory()

# def get_memory(session_id):
#     return memory

# chat_chain = prompt | llm | StrOutputParser()

# chat_with_message_history = RunnableWithMessageHistory(
#     chat_chain,
#     get_memory,
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )

# response = chat_with_message_history.invoke(
#     {
#         "context": current_weather,
#         "question": "Hi, I am at Watergate Bay. What is the surf like?"
#     },
#     config={"configurable": {"session_id": "none"}}
# )
# print(response)

# response = chat_with_message_history.invoke(
#     {
#         "context": current_weather,
#         "question": "Where I am?"
#     },
#     config={"configurable": {"session_id": "none"}}
# )
# print(response)

# --------------------------------------------------------
# Storing Conversation History
# --------------------------------------------------------


# connection to the neo4j database
# graph = Neo4jGraph(
#     url="bolt://54.211.218.230:7687",
#     username="neo4j",
#     password="radio-cases-darts"
# )

# # returning the graph schema
# print(graph.schema)

# # executing a query
# result = graph.query("""
# MATCH (m:Movie)
# RETURN m.title, m.plot, m.poster
# """)

# print(result)

# -------------------- Store conversation history --------------------

# generating a random session uuid for identification of the conversation
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

graph = Neo4jGraph(
    url="bolt://34.227.171.104:7687",
    username="neo4j",
    password="crowns-benches-cleansers"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


chat_chain = prompt | llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Bells", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while (question := input("> ")) != "exit":

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,

        },
        config={
            "configurable": {"session_id": SESSION_ID}
        }
    )

    print(response)
