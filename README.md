----->RAG-CHATBOT<------

operations:
1. Load/Read pdf.
2. splitting into chunks.
3. Loading Embedding LLM model from Huggingface.
4. Vectors storing(Cromadb).
5. Retrieve Data from Cromadb.
6. Load LLM model for Text generation.
7. Prompting.
8. chaining.


Note: 1. Open source LLM model for text-generation used from Huggingface through HuggingfaceEndpoint. Interated with langcahin.
Means that call llm model from huggingface through langcahin.
       2. Also used Embedding model from Huggingface.
       
specilization: 1. Chatbot using RAG.
               2. provides option to upload pdf.
               3. all above mentioned options are happend on those pdf.
               4. asked any thing related to the pdf file.
               
            

