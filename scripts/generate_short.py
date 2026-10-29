import csv, json, os, re, subprocess, sys, textwrap
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT, WORK = Path("output"), Path(".render-work")
W, H, FPS = 1080, 1920, 30

def run(args):
    subprocess.run(args, check=True)

def get_inputs():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    d = event.get("inputs", {})
    title = str(d.get("title", "Capital Danny Short")).strip()[:180]
    narration = str(d.get("narration", "")).strip()
    raw = str(d.get("scenes", "")).strip()
    if not narration or len(narration) > 6000 or len(raw) > 8000:
        raise ValueError("Narration is required; maximum lengths are 6,000 narration and 8,000 scene characters.")
    scenes = []
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"): continue
        p = [x.strip() for x in line.split("|", 2)]
        if len(p) != 3: raise ValueError(f"Use seconds | NASA search terms | caption: {line}")
        scenes.append({"seconds": max(2, min(20, float(p[0]))), "query": p[1][:180], "caption": p[2][:100]})
        if len(scenes) == 10: break
    if not scenes:
        scenes = [{"seconds":8,"query":"black hole","caption":"SPACE IS STRANGER THAN FICTION"}]
    return title or "Capital Danny Short", narration, scenes

def nasa_image(query, target):
    try:
        r = requests.get("https://images-api.nasa.gov/search", params={"q":query,"media_type":"image","page_size":10}, timeout=12)
        r.raise_for_status()
        items = r.json().get("collection",{}).get("items",[])
    except Exception as e:
        print("NASA search failed:", e); return None
    for item in items[:10]:
        meta = (item.get("data") or [{}])[0]
        links = item.get("links") or []
        url = next((x.get("href") for x in links if x.get("rel")=="preview"), None) or next((x.get("href") for x in links if x.get("href")),None)
        if not url: continue
        try:
            r = requests.get(url, timeout=15, stream=True); r.raise_for_status()
            if not r.headers.get("Content-Type","").lower().startswith("image/"): continue
            with target.open("wb") as f:
                for chunk in r.iter_content(131072):
                    if chunk: f.write(chunk)
            with Image.open(target) as im: im.verify()
            return str(meta.get("title","NASA image")), str(meta.get("photographer") or meta.get("center") or "NASA / source metadata"), url
        except Exception:
            target.unlink(missing_ok=True)
    return None

def fallback(path, caption):
    im=Image.new("RGB",(W,H),(5,10,26)); d=ImageDraw.Draw(im)
    for r in range(740,80,-20):
        s=max(8,34-r//35); d.ellipse((W//2-r,H//2-r,W//2+r,H//2+r),outline=(s,s+5,min(100,s+25)),width=3)
    cx,cy=W//2,H//2-100
    d.ellipse((cx-185,cy-185,cx+185,cy+185),fill=(0,0,0))
    d.ellipse((cx-300,cy-65,cx+300,cy+65),outline=(210,125,48),width=18)
    fp="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font=ImageFont.truetype(fp,42) if Path(fp).exists() else ImageFont.load_default()
    d.text((40,45),"CAPITAL DANNY  •  SCIENCE",font=font,fill=(220,230,250))
    y=H-500
    for line in textwrap.wrap(caption,width=25)[:3]:
        b=d.textbbox((0,0),line,font=font,stroke_width=2)
        d.text(((W-(b[2]-b[0]))/2,y),line,font=font,fill="white",stroke_width=2,stroke_fill="black"); y+=62
    im.save(path,quality=92)

def verticalize(src,dst):
    with Image.open(src) as im:
        im=ImageOps.exif_transpose(im).convert("RGB")
        ImageOps.fit(im,(W,H),method=Image.Resampling.LANCZOS).save(dst,quality=92)

def get_duration(path):
    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)],check=True,capture_output=True,text=True)
    return max(1,float(r.stdout.strip()))

def srt_time(seconds):
    ms=max(0,int(round(seconds*1000)))
    h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); sec,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def make_captions(audio, srt_path):
    # Word timestamps come from the actual synthesized audio, not scene durations.
    from faster_whisper import WhisperModel
    model=WhisperModel("tiny.en",device="cpu",compute_type="int8")
    segments,_=model.transcribe(str(audio),word_timestamps=True,vad_filter=True,beam_size=3)
    words=[]
    for segment in segments:
        for word in (segment.words or []):
            token=word.word.strip()
            if token and word.start is not None and word.end is not None:
                words.append((float(word.start),float(word.end),token))
    if not words:
        raise RuntimeError("Speech recognition produced no caption timestamps.")
    entries=[]
    for i in range(0,len(words),6):
        chunk=words[i:i+6]
        entries.append((chunk[0][0],chunk[-1][1]," ".join(w[2] for w in chunk).strip()))
    with srt_path.open("w",encoding="utf-8") as f:
        for i,(start,end,text) in enumerate(entries,1):
            safe=text.replace("-->","→")
            f.write(f"{i}\\n{srt_time(start)} --> {srt_time(max(start+0.12,end))}\\n{safe}\\n\\n")
    print(f"Created {len(entries)} synchronized caption cues from {len(words)} word timestamps.")

def main():
    title,narration,scenes=get_inputs()
    OUT.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file(): p.unlink()
    txt=WORK/"narration.txt"; txt.write_text(narration,encoding="utf-8")
    audio=WORK/"narration.wav"
    # Neural Piper voice sounds more natural than the old espeak-ng voice.
    run([sys.executable,"-m","piper","-m","en_US-lessac-medium","-f",str(audio),"--input-file",str(txt)])
    total=get_duration(audio)
    make_captions(audio,OUT/"captions.srt")
    wanted=sum(s["seconds"] for s in scenes)
    durations=[total*s["seconds"]/wanted for s in scenes]
    credits=[]; clips=[]; font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    for i,(s,secs) in enumerate(zip(scenes,durations),1):
        raw,img,clip=WORK/f"source_{i}.img",WORK/f"scene_{i}.jpg",WORK/f"clip_{i:02}.mp4"
        found=nasa_image(s["query"],raw) if s["query"] else None
        if found:
            try:
                verticalize(raw,img)
                credits.append({"scene":i,"query":s["query"],"asset_title":found[0],"credit":found[1],"source_url":found[2],"status":"Check item-specific rights/credits before use"})
            except Exception:
                fallback(img,""); credits.append({"scene":i,"query":s["query"],"asset_title":"","credit":"Original abstract fallback","source_url":"","status":"fallback"})
        else:
            fallback(img,""); credits.append({"scene":i,"query":s["query"],"asset_title":"","credit":"Original abstract fallback; no external image","source_url":"","status":"fallback"})
        vf=(f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"zoompan=z='min(zoom+0.00035,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS}")
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-loop","1","-framerate",str(FPS),"-i",str(img),"-vf",vf,"-t",f"{secs:.3f}","-an","-c:v","libx264","-preset","ultrafast","-crf","25","-pix_fmt","yuv420p","-r",str(FPS),"-movflags","+faststart",str(clip)])
        clips.append(clip)
    concat=WORK/"concat.txt"; concat.write_text("".join(f"file '{p.resolve()}'\n" for p in clips),encoding="utf-8")
    video=WORK/"video.mp4"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0","-i",str(concat),"-c","copy",str(video)])
    safe=re.sub(r"[^A-Za-z0-9_-]+","-",title).strip("-")[:70] or "capital-danny-short"
    final=OUT/f"{safe}.mp4"
    subtitle_filter=f"subtitles={str((OUT/'captions.srt').resolve())}:force_style='FontName=DejaVu Sans,FontSize=52,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,BackColour=&H90000000,Outline=2,Shadow=0,Alignment=2,MarginV=300'"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(video),"-i",str(audio),"-map","0:v:0","-map","1:a:0","-vf",subtitle_filter,"-c:v","libx264","-preset","ultrafast","-crf","24","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-shortest","-movflags","+faststart",str(final)])
    (OUT/"asset-credits.json").write_text(json.dumps(credits,indent=2,ensure_ascii=False),encoding="utf-8")
    with (OUT/"asset-credits.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["scene","query","asset_title","credit","source_url","status"]); w.writeheader(); w.writerows(credits)
    (OUT/"narration.txt").write_text(narration+"\n",encoding="utf-8")
    (OUT/"review-before-publishing.txt").write_text("REVIEW REQUIRED: no YouTube upload was performed. Check science, pronunciation, caption timing, output quality and each asset's item-level rights/credit notes. NASA search results are not automatic rights clearance.\n",encoding="utf-8")
    print(f"Rendered {final} ({total:.1f} seconds)")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print("ERROR:",e,file=sys.stderr); raise
