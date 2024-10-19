import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.prompts import *

print(RESEARCH_AGENT_PROMPT)