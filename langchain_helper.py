from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os
os.environ['OPENAI_API_KEY'] = 'sk-aIMdHPSLLG9AEsGuRsGRT3BlbkFJgNm1ZVBWbEc3wx'

llm = OpenAI(temperature=0.7)




def generate_restaurant_name_and_items(cuisine):
    #Chain 1: Restaurant Name
    promp_tempplate_name = PromptTemplate(
        input_variables=['cuisine'],
        template='I want to open a restaurant for {cuisine} food. Seggest a fancy name for this.'
    )
    
    name_chain = LLMChain(llm=llm, prompt=promp_tempplate_name, output_key='restaurant_name')
    
    #Chain 2: Menu Items
    promp_tempplate_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separate string"
    )
    
    food_item_chain = LLMChain(llm=llm,prompt=promp_tempplate_items, output_key="menu_items")
    
    chain = SequentialChain(
        chains=[name_chain,food_item_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name','menu_items']
    )
    
    response = chain({'cuisine': cuisine})
    
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))