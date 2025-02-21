import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Cấu hình API Key của Gemini
os.environ["GOOGLE_API_KEY"] = ""  # Thay bằng API key của bạn

# Đường dẫn database vector
vector_db_path = "E:/umbalaxibua/paragame_olympic/vectorstores/db_faiss"

# Load LLM từ API Gemini
def load_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.01)
    return llm

# Tạo prompt template
def create_prompt():
    template = """Bạn là một trợ lý AI thông minh, được sử dụng để trả lời câu hỏi dựa trên dữ liệu cung cấp.
Dưới đây là ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {question}

Hãy trả lời một cách đầy đủ và dễ hiểu bằng tiếng việt dựa trên thông tin có trong ngữ cảnh. Nếu không có thông tin phù hợp, hãy nói rằng bạn không biết.
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Đọc dữ liệu từ VectorDB
def read_vectors_db():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)

    return db

# Tạo chain truy vấn
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain


class Chatbot():
    def __init__(self):
        # Khởi tạo các thành phần
        self.db = read_vectors_db()
        self.llm = load_llm()
        self.prompt = create_prompt()
        self.llm_chain = create_qa_chain(self.prompt, self.llm, self.db)
    
    def answer(self, question):
        response = self.llm_chain.invoke({"query": question})
        return response


if __name__ == '__main__':
    # Khởi tạo các thành phần
    chatbot = Chatbot()

    # Chạy chain với câu hỏi
    question = "Why There Is No Documentation Phase"
    
    print(chatbot.answer(question=question))
