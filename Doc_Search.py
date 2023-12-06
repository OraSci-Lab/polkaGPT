 
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentType, initialize_agent   
class Doc_search():
    def __init__(self, llm, rust, polkadot, memory):
        self.llm=llm
        self.db_rust=rust
        self.db_polkadot=polkadot
        self.memory=memory
    def init_tool(self):
    
        rust_tool = create_retriever_tool(
        self.db_rust.as_retriever(search_kwargs={'k': 2}),
        name="RUSTGPT",
        description="Useful when you need to search for and return documents related to questions about Rust")
        
        polkadot_tool = create_retriever_tool(
        self.db_polkadot.as_retriever(search_kwargs={'k': 2}),
        name="POLKADOTGPT",
        description="Useful when you need to search for and return documents related to questions about Web3, Blockchain, Polkadot,...")

        return [rust_tool,polkadot_tool]

    def agents(self):
    # initialize conversational memory
       
        tools=Doc_search.init_tool(self)
        # initialize the agent
        agent_1 = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=self.llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        memory=self.memory)
        return agent_1

