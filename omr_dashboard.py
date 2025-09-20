# ==============================
# AUTOMATED OMR EVALUATION & SCORING SYSTEM
# ==============================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image

# -------------------------
# CREATE FOLDERS
# -------------------------
os.makedirs("./answer_keys", exist_ok=True)
os.makedirs("./results", exist_ok=True)

answer_key_file = "./answer_keys/setA.json"

# -------------------------
# CREATE OR LOAD ANSWER KEY
# -------------------------
if not os.path.exists(answer_key_file):
    answer_key = {
        "subject1": {"Q1":"A","Q2":"C","Q3":"C","Q4":"C","Q5":"C","Q6":"A","Q7":"C","Q8":"C",
                     "Q9":"B","Q10":"C","Q11":"A","Q12":"A","Q13":"D","Q14":"A","Q15":"B",
                     "Q16":"A,B,C,D","Q17":"C","Q18":"D","Q19":"A","Q20":"B"},
        "subject2": {"Q21":"A","Q22":"D","Q23":"B","Q24":"A","Q25":"C","Q26":"B","Q27":"A","Q28":"B",
                     "Q29":"D","Q30":"C","Q31":"C","Q32":"A","Q33":"B","Q34":"C","Q35":"A",
                     "Q36":"B","Q37":"D","Q38":"B","Q39":"A","Q40":"B"},
        "subject3": {"Q41":"C","Q42":"C","Q43":"C","Q44":"B","Q45":"B","Q46":"A","Q47":"C","Q48":"B",
                     "Q49":"D","Q50":"A","Q51":"C","Q52":"B","Q53":"C","Q54":"C","Q55":"A",
                     "Q56":"B","Q57":"B","Q58":"A","Q59":"A,B","Q60":"B"},
        "subject4": {"Q61":"B","Q62":"C","Q63":"A","Q64":"B","Q65":"C","Q66":"B","Q67":"B","Q68":"C",
                     "Q69":"C","Q70":"B","Q71":"B","Q72":"B","Q73":"D","Q74":"B","Q75":"A",
                     "Q76":"B","Q77":"B","Q78":"B","Q79":"B","Q80":"B"},
        "subject5": {"Q81":"A","Q82":"B","Q83":"C","Q84":"B","Q85":"C","Q86":"B","Q87":"B","Q88":"B",
                     "Q89":"A","Q90":"B","Q91":"C","Q92":"B","Q93":"C","Q94":"B","Q95":"B",
                     "Q96":"B","Q97":"C","Q98":"A","Q99":"B","Q100":"C"}
    }
    with open(answer_key_file,"w") as f:
        json.dump(answer_key,f, indent=4)
else:
    with open(answer_key_file,"r") as f:
        answer_key = json.load(f)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def detect_bubbles(thresh_img):
    contours,_ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        area=w*h
        aspect=w/float(h)
        if 400<area<2500 and 0.8<=aspect<=1.2:
            bubble_contours.append(cnt)
    bubble_contours=sorted(bubble_contours,key=lambda c: (cv2.boundingRect(c)[1],cv2.boundingRect(c)[0]))
    return bubble_contours

def is_filled(thresh_img,contour,fill_threshold=0.5):
    mask=np.zeros(thresh_img.shape,dtype="uint8")
    cv2.drawContours(mask,[contour],-1,255,-1)
    filled=cv2.countNonZero(cv2.bitwise_and(thresh_img,thresh_img,mask=mask))
    total=cv2.countNonZero(mask)
    return (filled/total)>fill_threshold if total>0 else False

def map_bubbles_to_answers(bubbles,thresh_img,options=4):
    answers={}
    question_num=1
    for i in range(0,len(bubbles),options):
        question_bubbles=bubbles[i:i+options]
        filled_options=[]
        for j,cnt in enumerate(question_bubbles):
            if is_filled(thresh_img,cnt):
                filled_options.append(chr(65+j))
        if len(filled_options)==1:
            answers[f"Q{question_num}"]=filled_options[0]
        elif len(filled_options)==0:
            answers[f"Q{question_num}"]="-"
        else:
            answers[f"Q{question_num}"] = ",".join(filled_options)
        question_num+=1
    return answers

def score_student(student_answers, answer_key):
    scores={}
    total=0
    for subject,q_dict in answer_key.items():
        sub_score=0
        for q,correct_ans in q_dict.items():
            student_ans=student_answers.get(q,"-")
            if "," in correct_ans:
                if student_ans in correct_ans.split(","):
                    sub_score+=1
            elif student_ans==correct_ans:
                sub_score+=1
        scores[subject]=sub_score
        total+=sub_score
    scores["total"]=total
    return scores

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("Automated OMR Evaluation & Scoring System")
st.write("Upload OMR sheet images to detect and score answers automatically.")

uploaded_files = st.file_uploader("Upload OMR images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    all_results=[]
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        img_cv = np.array(img)[:,:,::-1].copy()  # PIL to OpenCV BGR
        thresh = preprocess_image(img_cv)
        bubbles = detect_bubbles(thresh)
        student_answers = map_bubbles_to_answers(bubbles,thresh)
        scores = score_student(student_answers,answer_key)
        result={"student_id":uploaded_file.name.split(".")[0],**student_answers,**scores}
        all_results.append(result)
    
    df_results=pd.DataFrame(all_results)
    st.success("✅ Scoring completed!")
    st.dataframe(df_results)
    
    # Save CSV
    csv_file="./results/omr_scored_results.csv"
    df_results.to_csv(csv_file,index=False)
    st.download_button("Download CSV", data=open(csv_file,"rb"),file_name="omr_scored_results.csv")
