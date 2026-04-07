import os
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. Define transformation functions
# These functions take the input dictionary and extract what they need
def uppercase_text(data):
    # data is the dictionary passed from the previous step
    return data["input"].upper()

def count_words(data):
    # data is the dictionary passed from the previous step
    return len(data["input"].split())

# 2. Construct the Parallel Chain
# RunnableParallel runs all branches at the same time
complex_chain = RunnableParallel(
    # Branch 1: Pass the input exactly as it is
    original=RunnablePassthrough(),
    
    # Branch 2: Transform the input to uppercase
    shouting=uppercase_text,
    
    # Branch 3: Calculate word count
    word_count=count_words
)

# 3. Execution
if __name__ == "__main__":
    # The input to the chain is a dictionary
    input_data = {"input": "learning langchain is fun"}
    
    # Invoke the chain
    result = complex_chain.invoke(input_data)
    
    # 4. Results
    print("--- Chain Result ---")
    print(result)
    
    print("\n--- Breakdown ---")
    print(f"Original Input: {result['original']['input']}")
    print(f"Shouting Vers: {result['shouting']}")
    print(f"Total Words:   {result['word_count']}")