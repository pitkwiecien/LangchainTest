import config
from git_manager import GitManager
from file_manager import FileManager
from pinecone_manager import PineconeManager
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import StdOutCallbackHandler


file_contents = GitManager.get_files_content(config.PROJECT_REPO_LOCATION)
split_contents = FileManager.split_content(file_contents)
# FileManager.print_split_file_content(split_contents)

pinecone_mgr = PineconeManager(config.PINECONE_INDEX_NAME)
pinecone_mgr.index_content(split_contents)

index = pinecone_mgr.get_index()

query = "Add a parameter ram to the class Computer and return the whole class with the changes"
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=index.as_retriever(),
    callbacks=[StdOutCallbackHandler()],
    verbose=True
)

result = qa.run(query)
print(result)