from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

search = TavilySearchResults(
    max_results=3,
    description="Search the internet for current, real-time information and facts."
)
