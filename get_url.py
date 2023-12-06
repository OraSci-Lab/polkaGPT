import pandas as pd
from langchain.vectorstores import FAISS
import openai, os
from langchain.embeddings import  OpenAIEmbeddings
import openai, os
from langchain.tools import YouTubeSearchTool
from langchain.embeddings import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import ast, string
from YT_Search import YouTubeSearchTool

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPENAI_KEY')
openai.api_key = API_KEY ## To configure OpenAI API
os.environ["OPENAI_API_KEY"] = API_KEY ## To configure langchain con

 ## To configure langchain connections with OpenAI
class get_link():
    def get_and_save_transcript(video_id):
        try:    
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            transcript= {"start": [0], "text":["No transcript"]}
        return transcript
       

    # concat and limit length of documents in transcript (length <=150)
    def concat_text_and_time(df):
        concatenated_text = ''
        start_time = None
        list_text=[]
        list_time=[]
        for index, row in df.iterrows():
            if start_time is None:
                start_time = row['start']

            concatenated_text += row['text'] + ' '

            if len(concatenated_text.split()) >= 150:
                list_time.append(start_time)
                list_text.append(concatenated_text)
                concatenated_text = ''
                start_time = None
            elif index==df.shape[0]-1:
                list_time.append(start_time)
                list_text.append(concatenated_text)
        df=pd.DataFrame({'start': list_time, 'text': list_text})
        return df


    def preprocessing_link(list_link):
        list_link_output=[]
        for i in list_link:
            substring = "&pp"
            index = i.find(substring)
            if index != -1:
                result = i[:index]
                list_link_output.append(result)
        return list_link_output

    def ytb_search(query, number_video):
        tool = YouTubeSearchTool()
        output_YT=tool.run(query+"\n", number_video)
        list_link=output_YT['list_link']
        list_title=output_YT['list_title']
        list_link=get_link.preprocessing_link(list_link)
        output_link=[]
        count=1
        for i in range(len(list_link)):
            count+=1
            transcript=url(query+"\n",list_link[i].strip(string.punctuation),list_title[i]).get_df()
            #print(type(transcript), "    ", transcript.columns)
            if transcript['text'].values[0]=='No transcript':
                continue
            else:
                link=url(query+"\n",list_link[i].strip(string.punctuation),list_title[i]).output()
            output_link.append(link)
        return output_link

class url():
    def __init__(self, query,link, title):
        
        self.query=query
        self.path_video=link
        self.title=title
        

    def get_id_video(self):
        # find position of '?v=' in URL
        index = self.path_video.find('?v=')
        
        # if '?v=' not in URL, return None
        if index == -1:
            return None
        
        # Remove string from position of '?v=' to end of URL
        video_id = self.path_video[index + 3:]
        return video_id
    
    def get_df(self):
        # Create dataframe contain transcript
        video_id= url.get_id_video(self)
        transcript=get_link.get_and_save_transcript(video_id)
        df = pd.DataFrame(transcript)
        result_df = get_link.concat_text_and_time(df)
        result_df['title']=[self.title]*len(result_df)
        return result_df
    
    def get_db(df):
        transcript=df['text'].values
        titles=df['title'].values
        transcript_db = FAISS.from_texts(transcript,embedding)
        title_db =FAISS.from_texts(titles,embedding)
        return transcript_db,title_db
        
    def convert_seconds_to_hours_minutes(seconds):
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        return hours, minutes

    def find_time_for_text(self,df, doc, titles):
        # get the row with the highest score of a + b.  
        for i in range(len(df)):
            # Finding row contains search string 
            for j in range(len(doc)):
                if df.at[i,'text']==doc[j][0].page_content:
                    df.at[i,'score_total']=doc[j][1]+titles[i][1]
        matching_rows = df.loc[df['score_total'].idxmax()]
        # get time of above row
        time_values = matching_rows['start'].tolist()
        return time_values, matching_rows['score_total']
    
    def get_doc(self, df):
        transcript_db, title_db=url.get_db(df)
        title=title_db.similarity_search_with_relevance_scores(self.query, k=len(df))
        docs = transcript_db.similarity_search_with_relevance_scores(self.query, k=len(df))
        return docs, title
        
    # output of YT Search function
    def output(self):
        df=url.get_df(self)
        doc,titles=url.get_doc(self, df)
        seconds, score_total = url.find_time_for_text(self, df, doc, titles)
        return {"link":f"{self.path_video}&t={seconds}s","score": score_total}
    

    
