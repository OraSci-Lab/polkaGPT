from get_url import get_link
from Doc_Search import Doc_search
from operator import itemgetter
class Get_response():
    def __init__(self, query, llm,rust,polkadot, memory):
        self.agent_1=Doc_search(llm, rust,polkadot,memory).agents()(query)
        self.final_answer=self.agent_1['output']
        self.query=query
        

        
    def output(self): 
        print("===============================AGENT============================\n",self.agent_1,"\n===========================================================\n")
        if len(self.agent_1['intermediate_steps'])>0:
            if self.agent_1['intermediate_steps'][0][0].tool in ["RUSTGPT",'POLKADOTGPT']:
                output_dict=get_link.ytb_search(self.query, 5)
                #print(output_dict)
                # get 2  links with highest score 
                sorted_items = sorted(output_dict, key=itemgetter('score'), reverse=True)[:2]
                
                link_output_sorted=[]
                for items in sorted_items:
                    link_output_sorted.append(items['link'])
                #print("max:   ",link_output_sorted)
                self.final_answer=self.final_answer +f"\n \nYou can refer to links: \n\n {link_output_sorted}"
        return self.final_answer
    
    