############### prompts for analysis ##############

TEXTBOOK_RAG_PROMPT = """\n"You are a research assistant who can provide specific information on the book 'Textbook of Diabetes". You must only respond with information about those documents related to the request. Make sure every documents are covered."

Context:
{context}

Question:
{question}

Answer:
"""

PAPER_RAG_PROMPT = """\n"You are a research assistant who can provide specific information on the papers provided. You must only respond with information about those documents related to the request. Make sure every documents are covered."

Context:
{context}

Question:
{question}

Answer:
"""

RESEARCH_AGENT_PROMPT = """You are a research assistant who can provide specific information on the documents received. You must only respond with information about the documents related to the request. Make sure every documents are covered."""

QUERY_AGENT_PROMPT = """You are a research assistant who can search for the most relevant and up-to-date research paper using the semantic query tool."""

SUPERVISOR_PROMPT = ("You are a supervisor tasked with managing a conversation between the"
    " following workers:  InformationRetriever and PaperInformationRetriever. Given the following user request,"
    " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " You should never ask your team to do anything beyond research. They are not required to write content or posts."
    " You should only pass tasks to workers that are specifically research focused."
    " Ask maximum four requests. Make sure each workers are called at least once."
    " When finished, respond with FINISH.")

############### prompts for pubmed search #################

PUBMED_SYSTEM_MESSAGE = """Please execute the following steps in sequence:
    1. Use the PubMed search tool to search for papers.
    2. Retrieve the abstracts from the search results.
    3. Screen the abstracts based on the criteria provided by the user.
    4. Fetch full-text articles for all the papers that pass step 3. Store the full-text articles in the Qdrant vector database, 
        and extract the requested information for each article that passed step 3 from the full-text using the query provided by the user.
    5. Please provide a full summary at the end of the entire flow executed, detailing each paper's title, PMID, and the whole process/screening/reasoning for each paper. 
    The user will provide the search query, screening criteria, and the query for information extraction.
    Make sure you finish everything in one step before moving on to next step.
    Do not call more than one tool in one action."""
    
PUBMED_TOOL_PROMPT = "Search PubMed for research papers and retrieve abstracts. Pass the abstracts (returned results) to another tool."

SEMANTIC_SCREENING_PROMPT = """Screen PubMed abstracts based on semantic similarity to inclusion/exclusion criteria. Uses cosine similarity between abstracts and criteria. Requires 'abstracts' and 'screening criteria' as input.
    The 'abstracts' is a list of dictionary with keys as PMID, Title, Abstract.
    Output a similarity scores for each abstract and send the list of pmids that passed the screening to Fetch_Extract_Tool."""

PUMBED_RAG_PROMPT = """Fetch full-text articles based on PMIDs and store them in a Qdrant vector database.
    Then extract information based on user's query via Qdrant retriever using a RAG pipeline.
    Requires list of PMIDs and user query as input."""