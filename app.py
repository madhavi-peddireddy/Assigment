# Import required libraries
import streamlit as st
import os
from beyondllm import source, retrieve, embeddings, generator, llms

# Set environment variables for API keys
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]  # Hugging Face token

st.title("Video Insights Chat: Your Queries Answered! 📹💬")

# Load Data from YouTube Video
input_text = st.text_input("Enter your YouTube link 📺")

# Initialize the `data` variable to None
data = None

if input_text:
    try:
        data = source.fit(
            path=input_text,
            dtype="youtube",
            chunk_size=1024,
            chunk_overlap=0
        )
        st.write("Data successfully loaded!")
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")

# Ensure `data` is loaded before proceeding
if data is not None:
    # Ask for query input after data is successfully loaded
    query = st.text_input("Enter your Prompt💬")
    
    if query:  # Proceed only if the query is provided
        # Set Embedding Model
        model_name = 'BAAI/bge-small-en-v1.5'
        
        try:
            embed_model = embeddings.HuggingFaceEmbeddings(
                model_name=model_name
            )

            retriever = retrieve.auto_retriever(
                data=data,
                embed_model=embed_model,
                type="hybrid",
                mode="OR",
                top_k=3
            )

            # Retrieve relevant content based on query
            retrieved_nodes = retriever.retrieve(query)

        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")

        # Set Language Model
        try:
            llm = llms.GeminiModel(
                model_name="gemini-pro",
                google_api_key=os.environ['GOOGLE_API_KEY']  # Pass the correct API key
            )

            # Create RAG (Retrieval-Augmented Generation) pipeline
            pipeline = generator.Generate(
                question=query,
                retriever=retriever,
                llm=llm
            )

            # Generate response from pipeline
            response = pipeline.call()
            st.markdown(f"**Generated Response:** {response}")

            

            # Evaluate the pipeline's RAG triad
            evals = pipeline.get_rag_triad_evals()
            st.markdown(f"**RAG Triad evaluations:** {evals}")

        except Exception as e:
            st.error(f"An error occurred while setting the LLM or generating the response: {e}")
else:
    st.write("Please enter a valid YouTube link.")
