from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


loader = PyPDFLoader(r"C:\Users\hp\my_research_assistant\merc.pdf")

pages = loader.load_and_split()

texts = [page.page_content for page in pages]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(pages)

embedding_model = OllamaEmbeddings(model="llama2")
vector_store = FAISS.from_documents(chunks, embedding_model)

llm = OllamaLLM(model="llama2")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})
conversation_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


#date tool
def dates_tool(_: str) -> str:
    dates_prompt = "Extract all dates and times mentioned in the PDF document."
    chat_history_msgs = memory.load_memory_variables({})["chat_history"] 
    return conversation_chain({
        "question": dates_prompt,
        "chat_history": chat_history_msgs
    })


dates_tool = Tool(
    name="DocumentDates",
    func=dates_tool,
    description="Extract dates and times from the PDF document. Input should be a simple question or empty string."
)

#doc tool
def doc_qa_tool(query: str) -> str:
    chat_history = memory.load_memory_variables({})["chat_history"]
    return conversation_chain({
        "question": query,
        "chat_history": chat_history
    })

doc_tool = Tool(
    name="DocumentQA",
    func=doc_qa_tool,
    description="Answer questions based on the PDF document. Input should be a simple question string."
)


tools = [doc_tool, dates_tool]

prefix = """
You are an AI assistant. You have access ONLY to the following tools:

DocumentQA: Answer questions based on the PDF document.
DocumentSummary: Extract dates and times from the PDF document.

When you use a tool,You must ALWAYS respond with three parts in EXACT order:

Thought: <your reasoning here>
Action: <tool_name>
Action Input: <tool_input>

Do NOT skip any part.
Do NOT output anything else.

DO NOT include parentheses, quotes, greetings, or requests for documents.
Do NOT invent or use any tools other than DocumentQA and DocumentSummary.
Assume all documents are already loaded and available.
Do NOT add any extra explanations or user instructions.

When using tools, do not rephrase or change the user's query unnecessarily.
Use the user's exact question as the Action Input.

Never write any other sentence or line outside of these three required lines.
Always output ONLY one (and exactly one) Thought, one Action, and one Action Input, without repetition or extra commentary.


"""


tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

suffix = """
Begin!

{input}
{agent_scratchpad}
"""
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=prefix + "\n" + tool_descriptions + suffix,
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  
)

print("Ask questions about the document or request date info. Type 'exit' or 'quit' to end.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("BBYE")
        break
    response = agent_executor.invoke(user_input)
    print("Assistant:", response)


