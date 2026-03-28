import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import settings
from backend.services import groq_service

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = None
        self.vector_store_path = settings.FAISS_INDEX_PATH
        
        # Try to load existing
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load existing FAISS index: {e}")

    def process_pdf(self, file_path: str, source_name: str) -> int:
        """Process a PDF and update FAISS index"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(chunks, metadatas=metadatas)
            
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        
        return len(chunks)

    def query_pipeline(self, question: str):
        """Retrieve context and generate answer"""
        if self.vector_store is None:
            raise ValueError("No documents uploaded to vector store yet.")
            
        docs = self.vector_store.similarity_search_with_score(question, k=3)
        
        context = ""
        formatted_chunks = []
        for doc, score in docs:
            context += f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}\n\n"
            # In FAISS L2 distance, lower score is better (distance)
            formatted_chunks.append({
                "content": doc.page_content,
                "score": round(float(score), 4),
                "source": doc.metadata.get("source", "Unknown")
            })
            
        prompt = f"""
        Answer the following question based ONLY on the provided context. If the context does not contain the answer, say "I cannot answer this based on the provided documents."
        
        Context:
        {context}
        
        Question: {question}
        """
        
        if groq_service.client:
            response = groq_service.client.chat.completions.create(
                model=settings.STUDENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
        else:
            answer = "API key not configured."
            
        return answer, formatted_chunks

# Singleton instance
rag_pipeline = RAGPipeline()

def process_pdf(file_path: str, filename: str) -> int:
    return rag_pipeline.process_pdf(file_path, filename)

def query_pipeline(question: str):
    return rag_pipeline.query_pipeline(question)
