import os
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
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
        """Process a PDF into Semantic Markdown Chunks and update FAISS index"""
        # Perfect layout preservation: PDF to Markdown
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Read the raw Markdown and mathematically slice it precisely between explicit Title/Headers.
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        split_docs = markdown_splitter.split_text(md_text)
        
        chunks = []
        metadatas = []
        for i, doc in enumerate(split_docs):
            content = doc.page_content
            # Retain the semantic headers in the metadata + the source
            meta = doc.metadata.copy()
            meta["source"] = source_name
            meta["chunk_index"] = i
            
            # Extract header path like "Abstract" or "4. Experiments"
            header_context = " > ".join([v for k,v in doc.metadata.items() if "Header" in k])
            if header_context:
                content = f"[Section: {header_context}]\n" + content
                
            chunks.append(content)
            metadatas.append(meta)
            
        # Overwrite the vector store entirely to prevent duplicate chunks on re-upload
        self.vector_store = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
            
            
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        
        return len(chunks)

    def query_pipeline(self, question: str):
        """Retrieve context and generate answer"""
        if self.vector_store is None:
            raise ValueError("No documents uploaded to vector store yet.")
            
        # VERY IMPORTANT RAG FIX: 
        # If the user asks for a generic "summary", simple vector semantic search 
        # will retrieve the Reference section instead of the paper's actual content!
        # Instead, we will force the retrieval of the first few chunks (Abstract/Intro) 
        # by searching for them directly using the vector store's underlying docstore.
        
        is_summary_request = any(word in question.lower() for word in ['summarize', 'summary', 'overview', 'about this paper'])
        
        formatted_chunks = []
        context = ""
        
        if is_summary_request:
            # Manually extract the first 6 chunks of the document (usually Title, Abstract, and Introduction)
            doc_count = 0
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if doc.metadata.get("chunk_index", 999) < 6:
                    context += f"[Source: {doc.metadata.get('source')} (Intro/Abstract)]\n{doc.page_content}\n\n"
                    formatted_chunks.append({
                        "content": doc.page_content,
                        "score": 0.0, # Hand-picked
                        "source": doc.metadata.get("source", "Unknown")
                    })
                    doc_count += 1
                if doc_count >= 6: break
                
            prompt = f"""
            You are a helpful AI research assistant. The user has asked for a summary of the uploaded paper.
            I have extracted the Abstract and Introduction sections of the paper for you. 
            Write a high-quality, comprehensive summary of the paper's core topic and findings based on this context.
            
            Context (Abstract & Intro):
            {context}
            
            User's Request: {question}
            """
        else:
            # Standard semantic search for specific factual questions
            # Force inject chunk 0 (Title/Authors) so the model always knows what paper it is reading
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if doc.metadata.get("chunk_index") == 0:
                    context += f"[Source: {doc.metadata.get('source')} (Header/Metadata)]\n{doc.page_content}\n\n"
                    formatted_chunks.append({
                        "content": doc.page_content[:200] + "... (truncated metadata)",
                        "score": 0.0,
                        "source": doc.metadata.get("source", "Header Context")
                    })
                    break

            docs = self.vector_store.similarity_search_with_score(question, k=6)
            for doc, score in docs:
                context += f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}\n\n"
                formatted_chunks.append({
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                    "source": doc.metadata.get("source", "Unknown")
                })
                
            prompt = f"""
            You are a helpful research assistant. Use the following extracted context chunks from a research paper to synthesize the best possible answer to the user's question.
            NOTE: The "Header/Metadata" chunk contains the title and authors of the current paper. Other chunks may contain text from the References section (which list authors of OTHER papers). 
            Even if the information is fragmented, do your best to piece it together rather than refusing to answer.
            
            Context:
            {context}
            
            Question: {question}
            """
        
        import google.generativeai as genai
        
        answer = "No API keys configured."
        # Try Groq First
        if groq_service.client:
            try:
                response = groq_service.client.chat.completions.create(
                    model=settings.TEACHER_MODEL if is_summary_request else settings.STUDENT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content
            except Exception as e:
                if settings.GEMINI_API_KEY:
                    print(f"Groq API Failed, falling back to Gemini: {e}")
                    genai.configure(api_key=settings.GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-1.5-pro' if is_summary_request else 'gemini-1.5-flash-8b')
                    answer = model.generate_content(prompt).text
                else:
                    answer = f"Groq Error: {str(e)}. (Provide GEMINI_API_KEY for fallback)"
        elif settings.GEMINI_API_KEY:
            # Groq missing, fallback initialized immediately
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-pro' if is_summary_request else 'gemini-1.5-flash-8b')
            answer = model.generate_content(prompt).text
            
        return answer, formatted_chunks

# Singleton instance
rag_pipeline = RAGPipeline()

def process_pdf(file_path: str, filename: str) -> int:
    return rag_pipeline.process_pdf(file_path, filename)

def query_pipeline(question: str):
    return rag_pipeline.query_pipeline(question)
