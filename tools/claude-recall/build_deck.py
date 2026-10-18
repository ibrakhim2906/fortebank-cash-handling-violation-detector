"""Build the pitch deck from measured results. Run: uv run --with python-pptx==1.0.2 python tools/claude-recall/build_deck.py"""
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches,Pt

INK=RGBColor(0x14,0x1B,0x2D);MUTED=RGBColor(0x5B,0x64,0x78);ACCENT=RGBColor(0x0B,0x7A,0x5C);LIGHT=RGBColor(0xF4,0xF6,0xF9)
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
BLANK=prs.slide_layouts[6]


def text(slide,x,y,w,h,lines,size=20,color=INK,bold=False):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
    for i,line in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        run=p.add_run();run.text=line;run.font.size=Pt(size);run.font.color.rgb=color;run.font.bold=bold;p.space_after=Pt(8)
    return box


def slide(title,kicker=None):
    s=prs.slides.add_slide(BLANK)
    bg=s.background.fill;bg.solid();bg.fore_color.rgb=RGBColor(0xFF,0xFF,0xFF)
    if kicker:text(s,0.7,0.45,12,0.4,[kicker.upper()],13,ACCENT,True)
    text(s,0.7,0.8,12,1.2,[title],34,INK,True)
    return s


def table(s,x,y,w,rows,widths,size=16):
    shape=s.shapes.add_table(len(rows),len(rows[0]),Inches(x),Inches(y),Inches(w),Inches(0.5*len(rows)));t=shape.table
    for c,wd in enumerate(widths):t.columns[c].width=Inches(wd)
    for r,row in enumerate(rows):
        for c,val in enumerate(row):
            cell=t.cell(r,c);cell.text=str(val);para=cell.text_frame.paragraphs[0]
            para.runs[0].font.size=Pt(size);para.runs[0].font.bold=r==0;para.runs[0].font.color.rgb=RGBColor(0xFF,0xFF,0xFF) if r==0 else INK
            cell.fill.solid();cell.fill.fore_color.rgb=ACCENT if r==0 else (LIGHT if r%2 else RGBColor(0xFF,0xFF,0xFF))


s=slide('CashLint: local AI review of cash-desk video','ForteBank Datathon 2026')
text(s,0.7,2.3,11.5,3,['Upload a cash-desk recording, get a verdict, typed violations, timecodes,',
                       'evidence frames and the exact НБРК №231 clause, in about a minute.',
                       '','Open-weight model on our own GPU. No cloud AI, no face identification, works offline.'],22,MUTED)

s=slide('The problem','Why it matters')
text(s,0.7,2.1,11.5,4,['• Cash-desk procedure violations are found by people re-watching CCTV: slow and inconsistent.',
                       '• A useful tool must be local (bank footage never leaves), explainable, and cite the actual rule.',
                       '• It must never invent a violation it cannot see, or a legal norm that does not apply.'],22)

s=slide('How it works','Architecture')
text(s,0.7,2.0,12,4.5,['1. Frame sampling: 16 overview images with source timestamps',
                       '2. Qwen3-VL-32B-Instruct-AWQ sweep → typed candidate events (14 case types)',
                       '3. Dense focused confirmation of each proposal: rejects unsupported concerns',
                       '4. Storage veto: counter-examples for "cash/safe left out" cut false alarms 6 → 3',
                       '5. Rescue pass: phase-shifted second sweep re-checks missed and uncertain events',
                       '6. Result: verdict, type, timecode, evidence JPEGs, cited clause, human-review caveat',
                       '','FastAPI async jobs · vLLM on one A40 · model port never exposed · 185 s budget inside 5 min'],20)

s=slide('Measured results','Real runs, frozen evaluator')
table(s,0.7,2.0,11.9,[['Recipe','Data','Correct','TP / FN / FP','Timed ±2 s'],
                      ['Original 8B','DEV30','12/30','2 / 14 / 3','1/16'],
                      ['32B proposal policy','DEV30','20/30','12 / 4 / 6','5/16'],
                      ['+ storage veto (selected)','DEV30','23/30','12 / 4 / 3','5/16'],
                      ['+ rescue recall pass','DEV30','24/30','13 / 3 / 3','5/16'],
                      ['+ rescue recall pass','all 40','28/40','14 / 7 / 5','6/22']],[3.6,1.5,1.5,2.9,2.4])
text(s,0.7,5.5,12,1.2,['Every event matched in time has the correct type. Public API on all 40 videos: 0 failed jobs, slowest 73 s, within the 5-minute limit.',
                       'All 40 recordings were reviewed during development, so these are not independent test scores.'],15,MUTED)

s=slide('Rule grounding without invention','Rule grounding')
table(s,0.7,2.0,11.9,[['Violation type','НБРК №231 clause (exact quote in every result)'],
                      ['Cash left outside storage, safe left open','п.20 ч.2: accepted cash is kept in lockable storage'],
                      ['Counting without the client','п.20 ч.1: cash is accepted and counted in the client\'s presence'],
                      ['No note-by-note count / no ID check / no receipt','п.19 пп.2 / пп.1 / пп.6'],
                      ['Personal items in the cash unit','п.12']],[5.4,6.5],15)
text(s,0.7,5.2,12,1.4,['Official Adilet text, current revision, SHA-256 recorded per quote.',
                       'Door, keys, food and reading have no directly applicable clause, so we make no legal claim for them.'],15,MUTED)

s=slide('Local, licensed, deployable','Technical solution')
text(s,0.7,2.0,12,4.5,['• Network-denied GPU run passed on the selected recipe',
                       '• Licence verifier: all 238 installed packages + CPython + FFmpeg + model weights, 0 failures',
                       '  (open source, commercial use permitted; model weights Apache-2.0)',
                       '• Every experiment logged, including the ones we dropped',
                       '• 1234 automated tests; recipe pinned in deploy/mission-best.env'],20)

s=slide('What we learned and what is next','Honest limits')
text(s,0.7,2.0,12,4.5,['• Timing: lasting states (cash out, door open) are flagged when first visible, not when',
                       '  they become a violation. Two model-based re-timing attempts made it worse, so we kept the safer version.',
                       '• Omissions (no receipt, no ID check) are never inferred from frames we did not see.',
                       '• Next: person-departure cues for timing, longer focused clips for omissions,',
                       '  and validation on a new camera layout.'],20)

out=Path(__file__).resolve().parents[2]/'docs/CashLint-pitch.pptx';prs.save(out);print(out,out.stat().st_size)
