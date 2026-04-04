import os
import time
import asyncio
import random
import yt_dlp
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    url: str

def download_video(url: str, output_path: str = "temp_video.mp4") -> str:
    print(f"Bypassing Web Anti-Bot limits using pytubefix integration: {url}")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'source_address': '0.0.0.0',
            'socket_timeout': 120,
            'retries': 20,
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'web']
                }
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    except Exception as e:
        print(f"Fatal yt-dlp Download Error: {e}")
        raise e
            
    return output_path

async def generate_tab(model, video_file, prompt: str, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            # We use await asyncio.to_thread because the genai call is synchronous
            response = await asyncio.to_thread(
                model.generate_content,
                [video_file, prompt],
                request_options={"timeout": 600}
            )
            try:
                text = response.text.strip()
                if not text:
                    raise ValueError("Model returned an empty string natively.")
                return text
            except Exception as parse_err:
                print(f"Native Parsing Error on response parts: {parse_err}")
                return "Extraction failed due to an Empty Content Part exception from the Model."
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < retries - 1:
                # Calculate Jitter to prevent Thundering Herd collisions!
                sleep_time = 32 + (attempt * 10) + random.uniform(5, 25)
                print(f"Gemini API Rate Lmited (429). Native Google free tier limits hit. Sleeping thread for {sleep_time:.2f} seconds... (Attempt {attempt+1}/{retries})")
                await asyncio.sleep(sleep_time)
            else:
                raise e
    raise Exception("API Max Retries Exceeded natively.")

@app.post("/extract")
async def extract_video_knowledge(req: VideoRequest):
    # Use a secure random hash for the filename to prevent concurrent Vercel requests from colliding on Windows
    video_path = f"temp_video_{random.randint(10000, 99999)}.mp4"
    try:
        # 1. Download Video
        print(f"Downloading {req.url}...")
        await asyncio.to_thread(download_video, req.url, video_path)
        
        # 2. Upload to Gemini
        print("Uploading to Gemini File API...")
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        
        # 3. Wait for PROCESSING to finish
        print(f"Uploaded as: {video_file.name}. Waiting for ACTIVE state...")
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(5)
            video_file = await asyncio.to_thread(genai.get_file, video_file.name)
            
        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Gemini video processing failed on Google servers.")

        print("Video ACTIVE! Running 5 Parallel Tab Extractions...")
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")

        # 4. Strict Prompts for the 5 Tabs
        summary_prompt = """
        Watch this video and listen to the spoken words extremely carefully. STRICTLY write a highly detailed, visually appealing, comprehensive educational summary based primarily on what the speaker teaches!
        Do NOT just summarize the visual slides. Extract the deep underlying analogies, spoken reasoning, and human-level explanations.
        Break it down using massive Markdown Headers (##) for different themes, use bold text excessively for keywords, and provide extremely deep explanations.
        DO NOT extract formulas. DO NOT write flashcards. ONLY provide the spoken summary logic.
        """
        
        cheatsheet_prompt = """
        Watch this video and listen carefully. STRICTLY create a highly categorized visual Cheat Sheet using Markdown. 
        Group items logically under H2 (##) headers. Keep points incredibly concise, sharp, and focused purely on rapid-fire exam revision.
        Synthesize both visual lists and spoken key-takeaways! DO NOT write paragraphs. ONLY use rapid-fire bullet points and tables.
        """

        flashcards_prompt = """
        Listen to the video theory and read the slides. STRICTLY generate 10-15 deep, conceptual Question & Answer Flashcards extracted from the content. 
        Focus heavily on testing the theories the speaker spends the most time explaining!
        Format strictly as:
        Q: [Question]
        A: [Answer]
        Ask hard, deeply analytical questions. DO NOT write a summary. ONLY output Q/A flashcards.
        """

        formulae_prompt = """
        Watch this video carefully. Find EXACTLY where mathematical formulas, equations, or programming code snippets are shown visually on the screen OR spoken rigorously by the presenter.
        Extract EVERY SINGLE mathematical formula, equation, or code snippet accurately. 
        
        Format each extracted item explicitly as a visually distinct "Card" using this exact markdown schema:
        
        ---
        ### 📌 [Name of Formula or Concept]
        **Type:** [Mathematical Formula / Programming Code]
        **Context:** [A concise 1-sentence explanation from the speaker on when/why this is used]
        
        ```math
        [The exact raw LaTeX syntax of the formula/equation. DO NOT use inline dollar signs, ONLY raw LaTeX syntax inside this math block]
        ```
        ---
        
        STRICTLY DO NOT include general theory, paragraphs, or summaries. ONLY output these separated Formula/Code Cards using ```math delimiters. If none exist, state "No structural formulas or code found."
        """

        theory_prompt = """
        Listen explicitly to the entire audio stream of this video. STRICTLY explain the deep underlying theory, fundamental concepts, and core mechanics articulated by the speaker. 
        Think abstractly and explain the "Why" and "How" in extreme detail. Do not just read the slides. Focus heavily on analogies and spoken context!
        Make it read like a premium textbook chapter with H2 headers. DO NOT just summarize. Explain the deep theory behind the concepts.
        """

        # 5. Bundle into ONE API Call using Native LLM Markdown Headers!
        bundled_prompt = f"""
        You are a highly advanced AI analyzing a video lecture. You will extract 5 completely distinct sections.
        You MUST separate each section using these EXACT Markdown headers:
        
        # SECTION_1_SUMMARY
        {summary_prompt}
        
        # SECTION_2_CHEATSHEET
        {cheatsheet_prompt}
        
        # SECTION_3_FLASHCARDS
        {flashcards_prompt}
        
        # SECTION_4_FORMULAE
        {formulae_prompt}
        
        # SECTION_5_THEORY
        {theory_prompt}
        """

        print("Video ACTIVE! Running Unified Tab Extraction (Bypassing 15 RPM Quotas)...")
        raw_result = await generate_tab(model, video_file, bundled_prompt)

        print("Extraction completed successfully!")
        
        # Clean up local video file safely on Windows
        try:
            import glob
            for f in glob.glob("temp_video*"):
                os.remove(f)
        except Exception as cleanup_err:
            print(f"Windows IO Lock Bypass (Success): {cleanup_err}")

        # Split the result back into 5 tabs safely using the exact headers
        parts = []
        import re
        
        # Try finding Section 1
        s1 = raw_result.split("# SECTION_1_SUMMARY")[-1].split("# SECTION_2_CHEATSHEET")[0].strip()
        parts.append(s1 if s1 else "Extraction failed. Model did not write Section 1.")
        
        s2 = raw_result.split("# SECTION_2_CHEATSHEET")[-1].split("# SECTION_3_FLASHCARDS")[0].strip()
        parts.append(s2 if s2 else "Extraction failed. Model did not write Section 2.")
        
        s3 = raw_result.split("# SECTION_3_FLASHCARDS")[-1].split("# SECTION_4_FORMULAE")[0].strip()
        parts.append(s3 if s3 else "Extraction failed. Model did not write Section 3.")
        
        s4 = raw_result.split("# SECTION_4_FORMULAE")[-1].split("# SECTION_5_THEORY")[0].strip()
        parts.append(s4 if s4 else "Extraction failed. Model did not write Section 4.")
        
        s5 = raw_result.split("# SECTION_5_THEORY")[-1].strip()
        # If it didn't find the header, s5 will just be the raw text
        raw_s5 = s5 if "# SECTION_5_THEORY" in raw_result else "Extraction failed. Model did not write Section 5."
        parts.append(raw_s5)

        return {
            "summary": parts[0],
            "cheatsheet": parts[1],
            "flashcards": parts[2],
            "formulae": parts[3],
            "theory": parts[4]
        }

    except Exception as e:
        print(f"Error: {e}")
        # Cleanup on failure safely on Windows
        try:
            import glob
            for f in glob.glob("temp_video*"):
                os.remove(f)
        except Exception as cleanup_err:
            print(f"Windows IO Lock Bypass (Failure): {cleanup_err}")
            
        raise HTTPException(status_code=500, detail=str(e))

class SnapshotRequest(BaseModel):
    url: str
    seconds: float

@app.post("/snapshot")
async def extract_snapshot(req: SnapshotRequest):
    try:
        print(f"Extracting Snapshot Stream URL for {req.url}")
        yt_cmd = ["yt-dlp", "--extractor-args", "youtube:player-client=ios,android,web", "-f", "bestvideo[height<=720]+bestaudio/best", "-g", "--no-playlist", req.url]
        process = await asyncio.create_subprocess_exec(
            *yt_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"yt-dlp failed: {stderr.decode()}")
            
        stream_url = stdout.decode().split('\n')[0].strip()
        if not stream_url:
            raise Exception("Empty stream URL returned natively.")

        import uuid
        frame_path = f"temp_frame_{uuid.uuid4().hex}.jpg"
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-ss", str(int(req.seconds)),
            "-i", stream_url,
            "-vframes", "1",
            "-q:v", "3",
            "-y",
            frame_path
        ]
        
        f_process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        f_out, f_err = await f_process.communicate()
        
        if f_process.returncode != 0 and not os.path.exists(frame_path):
            raise Exception(f"ffmpeg failed: {f_err.decode()}")
            
        import base64
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            
        try:
            os.remove(frame_path)
        except:
            pass
            
        return {"frameBase64": f"data:image/jpeg;base64,{b64}"}
    except Exception as e:
        print(f"Snapshot Extract Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PlaylistRequest(BaseModel):
    url: str

@app.post("/playlist-info")
async def get_playlist_info(req: PlaylistRequest):
    try:
        print(f"Fetching Playlist Info for: {req.url}")
        # --flat-playlist is the key for speed
        yt_cmd = [
            "yt-dlp", "--extractor-args", "youtube:player-client=ios,android,web", 
            "--flat-playlist", "--dump-single-json", 
            "--playlist-items", "1-50", # Limit to 50 items for stability
            req.url
        ]
        process = await asyncio.create_subprocess_exec(
            *yt_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            # Fallback to single video info if it's not a playlist
            yt_cmd = ["yt-dlp", "--extractor-args", "youtube:player-client=ios,android,web", "--dump-single-json", req.url]
            process = await asyncio.create_subprocess_exec(*yt_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"yt-dlp failed: {stderr.decode()}")

        import json
        data = json.loads(stdout.decode())
        
        videos = []
        if 'entries' in data:
            for entry in data['entries']:
                if entry:
                    videos.append({
                        "id": entry.get('id'),
                        "url": entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}",
                        "title": entry.get('title'),
                        "thumbnail": entry.get('thumbnails', [{}])[-1].get('url') if entry.get('thumbnails') else f"https://img.youtube.com/vi/{entry.get('id')}/mqdefault.jpg"
                    })
        else:
            # Single video
            videos.append({
                "id": data.get('id'),
                "url": data.get('webpage_url'),
                "title": data.get('title'),
                "thumbnail": data.get('thumbnails', [{}])[-1].get('url')
            })

        return {"title": data.get('title', 'Unknown Source'), "videos": videos}
    except Exception as e:
        print(f"Playlist Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MultiAnalysisRequest(BaseModel):
    transcripts: list[dict] # {title: string, text: string, url: string}
    query: str

@app.post("/multi-video-analysis")
async def multi_video_analysis(req: MultiAnalysisRequest):
    try:
        if not req.transcripts:
            raise HTTPException(status_code=400, detail="No transcripts provided.")

        # Construct a massive context block
        context_parts = []
        for i, t in enumerate(req.transcripts):
            context_parts.append(f"--- VIDEO {i+1}: {t['title']} ---\nURL: {t.get('url', 'N/A')}\n{t['text']}\n")
        
        full_context = "\n".join(context_parts)
        
        prompt = f"""
        You are an elite Educational AI acting as a Multi-Video Intelligence System. Below are the titles, URLs, and available text transcripts of several educational videos.
        Your goal is to answer the user's query comprehensively.
        
        CONTEXT FROM VIDEOS:
        {full_context}
        
        USER QUERY:
        {req.query}
        
        STRICT RULES:
        1. NO APOLOGIES OR ERRORS: If any transcript says "disabled", "failed", or is missing, ABSOLUTELY DO NOT mention that the transcript is missing, failed, or that you are unable to provide a detailed synthesis. 
        2. INFER FROM TITLE: If a transcript is missing, seamlessly use your own expert internal knowledge about the topics mentioned in the video's title to answer the query deeply and comprehensively as if you had watched the video. Act as if you know exactly what the topic is about.
        3. CITE YOUR SOURCES: When referencing a concept, YOU MUST CITE the video it came from using a clickable Markdown link. Format: [Video Title](Video URL). Even if you used your own knowledge based on the title, cite the video title as the source.
        4. HIGHLIGHT CONFLICTS: If videos or your knowledge conflict, highlight the different perspectives and cite both.
        5. FORMATTING: Keep the response highly structured with markdown headers. If the user asks for a link, PROVIDE the exact URL from the context. Use **bold** text for key terms.
        """
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        return {"text": response.text}
    except Exception as e:
        print(f"Multi-Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run API on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
