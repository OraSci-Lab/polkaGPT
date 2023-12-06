"""
Adapted from https://github.com/venuv/langchain_yt_tools

CustomYTSearchTool searches YouTube videos related to a person
and returns a specified number of video URLs.
Input to this tool should be a comma separated list,
 - the first part contains a person name
 - and the second(optional) a number that is the
    maximum number of video results to return
 """
import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from youtube_search import YoutubeSearch

class YouTubeSearchTool(BaseTool):
    """Tool that queries YouTube."""

    name: str = "youtube_search"
    description: str = (
        "search for youtube videos associated with a person. "
        "the input to this tool should be a comma separated list, "
        "the first part contains a person name and the second a "
        "number that is the maximum number of video results "
        "to return aka num_results. the second part is optional"
    )

    def _search(self, person: str, num_results: int) -> str:
        results = YoutubeSearch(person, num_results).to_json()
        data = json.loads(results)
        url_suffix_list = [
            "https://www.youtube.com" + video["url_suffix"] for video in data["videos"]
        ]
        title_list = [video["title"] for video in data["videos"]
        ]
        return {"list_link":url_suffix_list,"list_title": title_list}

    def _run(self,query, run_manager: Optional[CallbackManagerForToolRun] = None, num_results=5) -> str:
        """Use the tool."""
        question = query
        num_results = num_results # number of link return to find max link
        return self._search(question, num_results)
    
