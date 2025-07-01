import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
import re
from dotenv import load_dotenv
import traceback
import os
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
st.set_page_config(page_title="Dataopsguru.com: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 Summarize Text From YT or Website")
st.subheader('Summarize URL')
generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
You are a professional summarizer. Summarize the following content clearly in **less than 300 words** without repeating phrases, filler text, or requests for more context. Use clear bullet points if appropriate. Return only the clean summary without any preamble, disclaimers, or repeated patterns.
Content:\n{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_video_id(url):
    pattern = r'(?:v=|\\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = get_video_id(generic_url)
                    if not video_id:
                        st.error("Invalid YouTube URL. Please ensure you provided a correct video URL.")
                        st.stop()

                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi', 'en'])
                        transcript_text = " ".join([t['text'] for t in transcript])
                        docs = [Document(page_content=transcript_text)]
                    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
                        st.error("This YouTube video does not have captions available in Hindi or English. Please try a different video.")
                        st.stop()
                    except VideoUnavailable:
                        st.error("This YouTube video is unavailable. It may be private, removed, or region-blocked.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error retrieving YouTube transcript: {e}")
                        st.text(traceback.format_exc())
                        st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                output_words = output_summary.split()
                if len(output_words) > 300:
                    output_summary = " ".join(output_words[:300]) + "..."
                st.success(output_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.text(traceback.format_exc())
