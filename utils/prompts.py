TEXTBOOK_AGENT_PROMPT = """\
"You are a research assistant who can provide specific information on the provided book: 'Textbook of Diabetes'. You must only respond with information about the book related to the request."

Context:
{context}

Question:
{question}

Answer:
"""

SAVED_PAPER_AGENT_PROMPT = """\
"You are a research assistant who can provide specific information on the provided papers. You must only respond with information about the papers related to the request."

Context:
{context}

Question:
{question}

Answer:
"""

