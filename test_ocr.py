import os
import sys
import time
import yt_dlp
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def download_video(url, output_path="video.mp4"):
    print(f"Downloading video from {url}...")
    # Cap resolution at 480p to keep downloads lightning fast while maintaining readability
    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/bestvideo[height<=480]+bestaudio/best',
        'outtmpl': output_path,
        'quiet': False,
        'overwrites': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def process_video_ocr(video_path):
    print(f"\nUploading {video_path} to Gemini File API...")
    video_file = genai.upload_file(path=video_path)
    print(f"Uploaded as URI: {video_file.name}. Waiting for Google servers to process the video stream...")
    
    # Wait for the file to finish processing (vision extraction)
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        print("\nVideo processing failed on Google servers.")
        return

    print(f"\nProcessing complete (State: {video_file.state.name}). Running OCR Extraction via gemini-2.5-flash...")
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    
    prompt = """
    Watch this video carefully from start to finish. You must rely on BOTH the spoken audio track AND the visual image frames on the screen.
    Pay extremely close attention to the visual text on the screen, chalkboard, or slides. If the speaker does not explicitly read a formula out loud, you MUST STILL EXTRACT IT by physically reading the screen.
    Extract EVERY SINGLE mathematical formula, code snippet, defined variable, and key theory.
    Format your output beautifully in Markdown. Use heavy bullet points, ## Markdown headers, ```code blocks``` for code, and structured Math equations for formulas.
    """
    
    print("\nGenerating...")
    # Add a generous timeout since video reading takes a few seconds longer than text
    start_time = time.time()
    response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
    end_time = time.time()
    
    print(f"\n--- TOTAL GENERATION TIME: {round(end_time - start_time, 2)}s ---\n")
    print(response.text)
    
    # Save purely for user to review
    with open("detailed_ocr_results.md", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("\nResults saved to detailed_ocr_results.md")

if __name__ == "__main__":
    # Default to a highly dense programming/math architecture video
    url = "https://www.youtube.com/watch?v=kYRB-vJVyRM" # Short React Hooks code tutorial
    if len(sys.argv) > 1:
        url = sys.argv[1]
        
    try:
        path = download_video(url)
        process_video_ocr(path)
    except Exception as e:
        print(f"Fatal error: {e}")
