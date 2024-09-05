import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # LLaMA 2 model initialization
        llm = CTransformers(
            model='model\\llama-2-7b-chat.ggmlv3.q8_0.bin',  # Ensure correct path format
            model_type='llama',
            config={
                'max_new_tokens': 256,
                'temperature': 0.01
            }
        )
        st.write("Model loaded successfully.")  # Debugging statement

        # Prompt Template
        template = """
            Write a blog for {blog_style} job profile on the topic "{input_text}"
            within {no_words} words.
        """
        
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", 'no_words'],
            template=template
        )
        
        # Generate the response from the LLaMA 2 model
        formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
        st.write(f"Formatted Prompt: {formatted_prompt}")  # Debugging statement
        
        response = llm.invoke(formatted_prompt)  # Use invoke instead of __call__
        st.write(f"Model Response: {response}")  # Debugging statement
        
        return response
    except Exception as e:
        st.write(f"Error generating response: {e}")
        return "Sorry, there was an error generating the blog. Please try again."

# Page configuration
st.set_page_config(
    page_title="Blog Generator",
    page_icon='✍️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Header
st.header("Generate Blogs ✍️")

# Input field for blog topic
input_text = st.text_input("Enter the Blog Topic", placeholder="Type your blog topic here...")

# Creating two columns for additional fields
col1, col2 = st.columns(2)

with col1:
    no_words = st.text_input('Number of Words', placeholder="e.g., 500, 1000")

with col2:
    blog_style = st.selectbox(
        'Writing Style',
        ('Researchers', 'Data Scientists', 'Common People'),
        index=0
    )

# Button to generate the blog
submit = st.button("Generate Blog")

# Final response
if submit:
    if input_text and no_words and blog_style:
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.write("Please fill in all fields before generating the blog.")
