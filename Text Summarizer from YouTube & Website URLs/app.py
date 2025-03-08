import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


## Streamlit App
st.set_page_config(page_title = "Langchain: Summarize Text From YT or Website", page_icon = "🐦‍🔥")
st.title("🐦‍🔥 Langchain: Summarize Text From YT ot Website")
st.subheader('Summarize URL')


## Get the Groq API Key and url (YT ot Website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value = "", type = "password")

generic_url = st.text_input("URL", label_visibility = "collapsed")

## Llama Model using Groq API
if not groq_api_key.strip():
    st.warning("Please enter your Groq API Key in the sidebar to proceed.")
else:
    try:
        llm = ChatGroq(model="Llama-3.3-70B-SpecDec", groq_api_key=groq_api_key)


        prompt_template = """
        Provide a summery of the following content in 500 words:
        Content: {text}

        """
        prompt = PromptTemplate(template = prompt_template, input_variables = ["text"])

        if st.button("Summarize the Content from YT or Website"):
            ## Validate all the inputs
            if not groq_api_key.strip() or not generic_url.strip():
                st.error("Please provide the information!")
            elif not validators.url(generic_url):
                st.error("Please enter a valid Url. It can only be a Yt or Website url")

            else:
                try:
                    with st.spinner("Waiting..."):
                        ## Loading the website or YT video data
                        if "youtube.com" in generic_url:
                            try:
                                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)  # Avoid fetching title
                                docs = loader.load()
                            except Exception as e:
                                st.error(f"Failed to load YouTube video content: {e}")
                        else:
                            loader = UnstructuredURLLoader(
                                urls = [generic_url], 
                                ssl_varify = False, 
                                headers = {
                                    "User-Agent": (
                                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/120.0.0.0 Safari/537.36 "
                                        "Edg/120.0.0.0 "
                                        "Firefox/120.0"
                                    )
                                }
                            )
                            ## So that all these browsers are supported
                        docs = loader.load()

                        ## Chain For Summarization
                        summary_chain = load_summarize_chain(
                            llm = llm,
                            chain_type = "stuff",
                            prompt = prompt
                        )
                        output_summary = summary_chain.run(docs)

                        st.success(output_summary)
                except Exception as e:
                    st.exception(f"Exception: {e}")
    except Exception as e:
        st.error(f"Failed to initialize Groq API: {e}")