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
import requests
import time

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

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
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript_robust(video_id, max_retries=3):
    """
    Robust YouTube transcript retrieval with multiple fallback strategies
    """
    # Extended language list - try more languages
    language_codes = [
        'en', 'hi',  # Primary languages
        'en-US', 'en-GB', 'en-CA', 'en-AU',  # English variants
        'hi-IN',  # Hindi variant
        'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',  # Other common languages
        'ar', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa',  # Indian languages
        'auto'  # Auto-generated if available
    ]
    
    for attempt in range(max_retries):
        try:
            # Method 1: Try with specific language preferences
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=language_codes[:10]  # Try first 10 languages
                )
                transcript_text = " ".join([t['text'] for t in transcript])
                return transcript_text, "success"
            except (TranscriptsDisabled, NoTranscriptFound):
                pass
            
            # Method 2: Try to get any available transcript
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # First, try manually created transcripts
                for transcript in transcript_list:
                    if not transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            transcript_text = " ".join([t['text'] for t in transcript_data])
                            return transcript_text, "manual_transcript"
                        except:
                            continue
                
                # Then try auto-generated transcripts
                for transcript in transcript_list:
                    if transcript.is_generated:
                        try:
                            transcript_data = transcript.fetch()
                            transcript_text = " ".join([t['text'] for t in transcript_data])
                            return transcript_text, "auto_generated"
                        except:
                            continue
                            
            except Exception as e:
                st.warning(f"Attempt {attempt + 1}: {str(e)}")
                
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2)
                
        except VideoUnavailable:
            return None, "video_unavailable"
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"error: {str(e)}"
            time.sleep(2)
    
    return None, "no_transcripts_found"

def check_video_accessibility(video_id):
    """Check if video is accessible"""
    try:
        # Simple check using YouTube's oembed API
        response = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=10)
        return response.status_code == 200
    except:
        return True  # Assume accessible if check fails

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

                    # Check video accessibility first
                    st.info("Checking video accessibility...")
                    if not check_video_accessibility(video_id):
                        st.error("This YouTube video appears to be unavailable or private.")
                        st.stop()

                    # Try robust transcript retrieval
                    st.info("Attempting to retrieve transcript...")
                    transcript_text, status = get_youtube_transcript_robust(video_id)
                    
                    if transcript_text:
                        if status == "auto_generated":
                            st.info("Using auto-generated captions (may be less accurate)")
                        elif status == "manual_transcript":
                            st.info("Using manual captions")
                        
                        docs = [Document(page_content=transcript_text)]
                    else:
                        # Provide more specific error messages
                        if status == "video_unavailable":
                            st.error("This YouTube video is unavailable. It may be private, removed, or region-blocked.")
                        elif status == "no_transcripts_found":
                            st.error("No captions/transcripts are available for this video. This could be because:")
                            st.markdown("""
                            - The video doesn't have captions enabled
                            - The video is too new (captions may be processing)
                            - The video is in a language not supported
                            - Regional restrictions apply
                            
                            Please try a different video with captions enabled.
                            """)
                        else:
                            st.error(f"Error retrieving transcript: {status}")
                        st.stop()
                        
                else:
                    # Website URL processing
                    st.info("Loading website content...")
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()

                # Generate summary
                st.info("Generating summary...")
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                
                # Ensure word limit
                output_words = output_summary.split()
                if len(output_words) > 300:
                    output_summary = " ".join(output_words[:300]) + "..."
                
                st.success("Summary generated successfully!")
                st.markdown("### Summary:")
                st.write(output_summary)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            with st.expander("View detailed error"):
                st.text(traceback.format_exc())
