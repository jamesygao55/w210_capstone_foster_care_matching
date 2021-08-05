"""
Streamlit Cheat Sheet
App to summarise streamlit docs v0.81.0 for quick reference
There is also an accompanying png version
https://github.com/daniellewisDL/streamlit-cheat-sheet
v0.71.0 November 2020 Daniel Lewis and Austin Chen
"""

from pathlib import Path
import base64

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from xgboost import XGBClassifier, XGBRegressor
import torch as torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import FMModel as FMModel
import DurationModel as DurationModel

from altair.vegalite.v4.schema.channels import X
import altair as alt
import geopandas as gpd
from PIL import Image

# Initial page config

st.set_page_config(
     page_title='Foster Care Matcher',
     layout="wide",
     initial_sidebar_state="expanded",
)

##########################
# htmls
##########################

HTML_WRAPPER1 = """<div style="background-color: #EAFAF1; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER2 = """<div style="background-color: #ededed; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
#HTML_WRAPPER3 = """<div style="background-color: #FEF9E7; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
HTML_WRAPPER3 = """<div style="background-color: #EAFAF1; overflow-x: auto; border: 1px solid #a6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
WHITECOLOR= '<p style="font-family:Courier; color:White; font-size: 20px;">....</p>'
WHITECOLORsmall= '<p style="font-family:Courier; color:White; font-size: 11px;">....</p>'
BANNER= '<p style="font-family:Helvetica Neue; color:Teal; font-size: 55px; line-height:25px;text-align: center;"><b>Foster Care Matcher</b></p>'
BANNERsmall= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: center;">Love. Heal. Respect. Cherish.</p>'
BANNERleft= '<p style="font-family:Helvetica Neue; color:Teal; font-size: 55px; line-height:25px;text-align: left;"><b>Foster Care Matcher</b></p>'
BANNERleftsmall= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: left;">Love. Heal. Respect. Cherish</p>'
SIDEBARHEADING= '<p style="font-family:Arial; color:Teal; font-size: 20px;text-align: left;"><b>Foster Care Matcher</b></p>'

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def set_session_resetter():
	st.session_state['resetter'] = False
	

def main():
    my_page = cs_sidebar()
    if my_page == 'Home': 
    	cs_home()
    elif my_page == 'Matcher':
    	cs_body()
    elif my_page == 'Journey':
    	cs_journey()
    elif my_page == 'Architecture':
    	cs_architecture()
    elif my_page == 'Team':
    	cs_team()
    return None

# Thanks to streamlitopedia for the following code snippet

# sidebar

def cs_sidebar():
	#st.markdown( """ <style> .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; } </style> """, unsafe_allow_html=True, )
	st.sidebar.write(SIDEBARHEADING,unsafe_allow_html=True)
	#set_png_as_page_bg('father-and-daughters-hand.jpeg')
#	header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#	img_to_bytes("father-and-daughters-hand.jpeg")
#	)
#	st.sidebar.markdown(
#	header_html, unsafe_allow_html=True,
#	)
#	image = Image.open('TopBanner6.png')
#	st.image(image,  width=600 ) #
#	st.write(BANNER,unsafe_allow_html=True) 
#	st.write(BANNERsmall,unsafe_allow_html=True) 


	mypage = st.sidebar.radio(' ', ['Home', 'Matcher', 'Journey', 'Architecture', 'Team'])

	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')
	st.sidebar.title('')

	st.sidebar.text('Summer 2021 Capstone')
	st.sidebar.text('School of Information')
	st.sidebar.text('University of California, Berkeley')

	return mypage


def cs_body():
    st.write(BANNERleft,unsafe_allow_html=True) 
    st.write(BANNERleftsmall,unsafe_allow_html=True) 

    col1, col2, col3 = st.beta_columns(3)

    placed_before = 'Select one'
    num_prev_placements = 0
    child_num_prev_placements_good = 0
    child_num_prev_placements_bad = 0
    child_date_of_first_placement = datetime.date(2015,1,1)
    child_recent_placement_outcome = 'Select one'
    child_ctkfamst = 'Select one'
    child_caretakerage = float("Nan")
    child_hispanic = 'Select one'
    child_mr_flag = False
    child_vishear_flag = False
    child_phydis_flag = False
    child_emotdist_flag = False
    child_othermed_flag = False
    child_clindis = 'Select one'
    child_everadpt = 'Select one'
    child_everadpt_age = float("Nan")
    current_case_goal = 'Select one'
    find_providers_button = None
#    ## need to add this so session state if resetter in session state blah blah blah
    if 'resetter' not in st.session_state:
        st.session_state['resetter'] = False

    col1.header("Child Information")
    col2.write(WHITECOLORsmall, unsafe_allow_html=True)
    col2.write(WHITECOLORsmall, unsafe_allow_html=True)

    child_birthday = col1.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now(),on_change = set_session_resetter)
    child_gender = col2.selectbox("Child's gender", ['Select one', 'Male', 'Female'], on_change = set_session_resetter)
    child_race = col1.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'], on_change = set_session_resetter)
    child_hispanic = col2.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)
    child_caretakerage = col1.number_input("Primary caretaker's age at the time of child's removal", min_value = 0, max_value = 100, step = 1,on_change = set_session_resetter)
    child_ctkfamst = col2.selectbox("What kind of caretaker was the child removed from?", ['Select one', 'Married Couple', 'Unmarried Couple', 'Single Female', 'Single Male'],on_change = set_session_resetter)


    if child_ctkfamst != 'Select one':
        col1.header("Prior Placement Information")
        col2.write(WHITECOLORsmall, unsafe_allow_html=True)
        col2.write(WHITECOLORsmall, unsafe_allow_html=True)
        placed_before = col1.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'], on_change = set_session_resetter)

        if placed_before == 'Yes':
             num_prev_placements = col2.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1, on_change = set_session_resetter)
        else:
            col2.subheader(" ")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
 
        if num_prev_placements > 0:
            child_num_prev_placements_good = col1.number_input('Previous placements with POSITIVE outcome', min_value = 0, max_value = num_prev_placements, step = 1,on_change = set_session_resetter)
            child_num_prev_placements_bad = col2.number_input('Previous placements with NEGATIVE outcome', min_value = 0, max_value = num_prev_placements, step = 1,on_change = set_session_resetter)

            child_date_of_first_placement = col1.date_input("First Placement Start Date", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now(),on_change = set_session_resetter)
            child_recent_placement_outcome = col2.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'],on_change = set_session_resetter)

        if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
            child_iswaiting = col1.selectbox("Is the child currently waiting for adoption?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)
            child_everadpt = col2.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'],on_change = set_session_resetter)

        if child_everadpt == 'Yes':
            child_everadpt_age = col1.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18,on_change = set_session_resetter)
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.subheader("")
   
                
    if  child_everadpt != 'Select one':

        col1.text("")
        col1.header("Disability Information")
        col2.write(WHITECOLORsmall,unsafe_allow_html=True) 
        col2.write(WHITECOLORsmall,unsafe_allow_html=True) 
        child_clindis = col1.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'],on_change = set_session_resetter)

        if child_clindis == 'Yes':
            col2.text("")
            col2.write("Check all that apply:")

            child_phydis_flag = col2.checkbox("Physically Disabled",on_change = set_session_resetter)
            child_vishear_flag = col2.checkbox("Visually or Hearing Impaired",on_change = set_session_resetter)
            col1.text("")
            col1.text("")
            child_mr_flag = col2.checkbox("Mental Retardation",on_change = set_session_resetter)
            child_emotdist_flag = col2.checkbox("Emotionally Disturbed",on_change = set_session_resetter)
            child_othermed_flag = col2.checkbox("Other Medically Diagnosed Condition",on_change = set_session_resetter)
            col2.text("")
                
            
        if ((child_clindis == 'Yes' and (child_mr_flag or child_vishear_flag or child_phydis_flag or child_emotdist_flag or child_othermed_flag))
            or  child_clindis == 'No' or child_clindis == 'Not yet determined'):
        
            if child_clindis !='Yes':
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("")
                col2.text("") 
                col2.text("")
                col2.text("")
                col2.text("")
     
            else:   
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col1.text("")
                col2.text("")
     

            col1.header("Removal Reasons")
            col2.write(WHITECOLOR, unsafe_allow_html=True)
            col2.write(WHITECOLORsmall,unsafe_allow_html=True)
            col1.write("Why did the child enter the foster care system? (Check all that apply)")
    

            physical_abuse = col1.checkbox('Physical Abuse',on_change = set_session_resetter)
            sexual_abuse = col1.checkbox('Sexual Abuse',on_change = set_session_resetter)
            emotional_abuse_neglect = col1.checkbox('Emotional Abuse',on_change = set_session_resetter)
            physical_neglect = col1.checkbox("Physical Neglect")
            medical_neglect = col1.checkbox("Medical Neglect",on_change = set_session_resetter)
            alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse",on_change = set_session_resetter)
            drug_abuse_child = col1.checkbox("Child's Drug Abuse",on_change = set_session_resetter)
            child_behavior_problem = col1.checkbox('Child Behavior Problem',on_change = set_session_resetter)
            child_disability = col1.checkbox('Child Disability',on_change = set_session_resetter)
            transition_to_independence = col1.checkbox("Transition to Independence",on_change = set_session_resetter)
            inadequate_supervision = col1.checkbox("Inadequate Supervision",on_change = set_session_resetter)
            adoption_dissolution = col1.checkbox("Adoption Dissolution",on_change = set_session_resetter)
            abandonment = col1.checkbox("Abandonment",on_change = set_session_resetter)
            labor_trafficking = col1.checkbox("Labor Trafficking")
            sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation",on_change = set_session_resetter)

            prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse",on_change = set_session_resetter)
            prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse',on_change = set_session_resetter)
            prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse",on_change = set_session_resetter)
            prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect',on_change = set_session_resetter)
            prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect",on_change = set_session_resetter)
            alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse",on_change = set_session_resetter)
            drug_abuse_parent = col2.checkbox("Parent's Drug Abuse",on_change = set_session_resetter)
            incarceration_of_parent = col2.checkbox('Incarceration of Parent',on_change = set_session_resetter)
            death_of_parent = col2.checkbox('Death of Parent',on_change = set_session_resetter)
            domestic_violence = col2.checkbox("Domestic Violence",on_change = set_session_resetter)
            inadequate_housing = col2.checkbox("Inadequate Housing",on_change = set_session_resetter)
            caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope",on_change = set_session_resetter)
            relinquishment = col2.checkbox('Relinquishment',on_change = set_session_resetter)
            request_for_service = col2.checkbox('Request for Service',on_change = set_session_resetter)
            csec = col2.checkbox("CSEC",on_change = set_session_resetter)


            col1.header("Current placement information")
            current_case_goal = col1.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'],on_change = set_session_resetter)
        
        if current_case_goal != 'Select one':
            col1.text("")
            col1.write("Current placement's applicable payments",on_change = set_session_resetter)
            current_case_ivefc = col1.checkbox("Foster Care Payments",on_change = set_session_resetter)
            current_case_iveaa = col1.checkbox("Adoption Assistance",on_change = set_session_resetter)
            current_case_ivaafdc = col1.checkbox("TANF Payment (Temporary Assistance for Needy Families)",on_change = set_session_resetter)
            current_case_ivdchsup = col1.checkbox("Child Support Funds",on_change = set_session_resetter)
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("") 
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            col2.text("")
            current_case_xixmedcd = col2.checkbox("Medicaid",on_change = set_session_resetter)
            current_case_ssiother = col2.checkbox("SSI or Social Security Benefits",on_change = set_session_resetter)
            current_case_noa = col2.checkbox("Only State or Other Support",on_change = set_session_resetter)
            current_case_payments_none = col2.checkbox("None of the above apply",on_change = set_session_resetter)
            current_case_fcmntpay = col1.number_input("Monthly Foster Care Payment ($)", min_value = 200, step = 100,on_change = set_session_resetter)
            col1.text("")
            col1.text("")
            
            find_providers_button = st.button("Find Providers")

#        ## Once the button is pressed, the resetter will be set to True and will be updated in the Session State
#        ## Recommender System output    
        if find_providers_button:
            if child_gender == 'Select one' or child_race == 'Select one' or child_clindis == 'Select one':
                st.error('Please fill in child\'s gender and race')
            else:
                st.session_state['resetter'] = True
			
            ## construct child record using user_input
        if st.session_state['resetter'] == True:
            child_input_record_data = {
            'PHYSICAL_ABUSE':[1.0 if physical_abuse else 0.0]
            ,'SEXUAL_ABUSE':[1.0 if sexual_abuse else 0.0]
            ,'EMOTIONAL_ABUSE_NEGLECT':[1.0 if emotional_abuse_neglect else 0.0]
            ,'ALCOHOL_ABUSE_CHILD':[1.0 if alcohol_abuse_child else 0.0]
            ,'DRUG_ABUSE_CHILD':[1.0 if drug_abuse_child else 0.0]
            ,'ALCOHOL_ABUSE_PARENT':[1.0 if alcohol_abuse_parent else 0.0]
            ,'DRUG_ABUSE_PARENT':[1.0 if drug_abuse_parent else 0.0]
            ,'PHYSICAL_NEGLECT':[1.0 if physical_neglect else 0.0]
            ,'DOMESTIC_VIOLENCE':[1.0 if domestic_violence else 0.0]
            ,'INADEQUATE_HOUSING':[1.0 if inadequate_housing else 0.0]
            ,'CHILD_BEHAVIOR_PROBLEM':[1.0 if child_behavior_problem else 0.0]
            ,'CHILD_DISABILITY':[1.0 if child_disability else 0.0]
            ,'INCARCERATION_OF_PARENT':[1.0 if incarceration_of_parent else 0.0]
            ,'DEATH_OF_PARENT':[1.0 if death_of_parent else 0.0]
            ,'CAREGIVER_INABILITY_TO_COPE':[1.0 if caregiver_inability_to_cope else 0.0]
            ,'ABANDONMENT':[1.0 if abandonment else 0.0]
            ,'TRANSITION_TO_INDEPENDENCE':[1.0 if transition_to_independence else 0.0]
            ,'INADEQUATE_SUPERVISION':[1.0 if inadequate_supervision else 0.0]
            ,'PROSPECTIVE_EMOTIONAL_ABUSE_NEGLECT':[1.0 if prospective_emotional_abuse_neglect else 0.0]
            ,'PROSPECTIVE_MEDICAL_NEGLECT':[1.0 if prospective_medical_neglect else 0.0]
            ,'PROSPECTIVE_PHYSICAL_ABUSE':[1.0 if prospective_physical_abuse else 0.0]
            ,'PROSPECTIVE_PHYSICAL_NEGLECT':[1.0 if prospective_physical_neglect else 0.0]
            ,'PROSPECTIVE_SEXUAL_ABUSE':[1.0 if prospective_sexual_abuse else 0.0]
            ,'RELINQUISHMENT':[1.0 if relinquishment else 0.0]
            ,'REQUEST_FOR_SERVICE':[1.0 if request_for_service else 0.0]
            ,'ADOPTION_DISSOLUTION':[1.0 if adoption_dissolution else 0.0]
            ,'MEDICAL_NEGLECT':[1.0 if medical_neglect else 0.0]
            ,'CSEC':[1.0 if csec else 0.0]
            ,'LABOR_TRAFFICKING':[1.0 if labor_trafficking else 0.0]
            ,'SEXUAL_ABUSE_SEXUAL_EXPLOITATION':[1.0 if sexual_abuse_sexual_exploitation else 0.0]
            ,'RACE_WHITE':[1.0 if child_race == 'White' else 0.0]
            ,'RACE_BLACK':[1.0 if child_race == 'Black' else 0.0]
            ,'RACE_ASIAN':[1.0 if child_race == 'Asian' else 0.0]
            ,'RACE_UNKNOWN':[0.0]
            ,'RACE_HAWAIIAN':[1.0 if child_race == 'Pacific Islander' else 0.0]
            ,'RACE_AMERICAN_INDIAN':[1.0 if child_race == 'Native American' else 0.0]
            ,'RACE_MULTI_RCL':[1.0 if child_race == 'Multi-Racial' else 0.0]
            ,'HISPANIC':[1.0 if child_hispanic == 'Yes' else 2.0]
            ,'AGE_AT_PLACEMENT_BEGIN':[round((datetime.datetime.date(datetime.datetime.now()) - child_birthday).days / 365, 2)]
            ,'NEW_REMOVAL':[1.0 if placed_before == 'Yes' else 0.0]
            #     #,REMOVAL_LENGTH #Need to make reflective as of placement begin date   
            #     #,PLACEMENT_NUMBER #Need to apply after using James's flattened version
            ,'CHILD_NUM_PREV_PLACEMENTS':[float(num_prev_placements)]
            ,'CHILD_NUM_PREV_PLACEMENTS_GOOD':[float(child_num_prev_placements_good)]
            ,'CHILD_NUM_PREV_PLACEMENTS_NEUTRAL':[max(float(num_prev_placements - child_num_prev_placements_good - child_num_prev_placements_bad), 0)]
            ,'CHILD_NUM_PREV_PLACEMENTS_BAD':[float(child_num_prev_placements_bad)]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_1.0':[1 if child_recent_placement_outcome == 'Positive' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_2.0':[1 if child_recent_placement_outcome == 'Neutral' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_3.0':[1 if child_recent_placement_outcome == 'Negative' else 0]
            ,'CHILD_PREV_PLACEMENT_OUTCOME_nan':[0]
            ,'CHILD_DAYS_SINCE_FIRST_PLACEMENT':[float((datetime.datetime.date(datetime.datetime.now()) - child_date_of_first_placement).days)]
            ,'CHILD_NUM_PREV_PLACEMENTS_GOOD_PERC':[round(child_num_prev_placements_good / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            ,'CHILD_NUM_PREV_PLACEMENTS_NEUTRAL_PERC':[round(max(float(num_prev_placements - child_num_prev_placements_good - child_num_prev_placements_bad), 0) / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            ,'CHILD_NUM_PREV_PLACEMENTS_BAD_PERC':[round(child_num_prev_placements_bad / float("Nan") if num_prev_placements == 0 else num_prev_placements,6)]
            #     #,'MOVE_MILES'
            #     #,'ROOMMATE_COUNT'
            ,'IVEFC':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivefc else 0.0)]
            ,'IVEAA':[float("Nan") if current_case_payments_none else (1.0 if current_case_iveaa else 0.0)]
            ,'IVAAFDC':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivaafdc else 0.0)]
            ,'IVDCHSUP':[float("Nan") if current_case_payments_none else (1.0 if current_case_ivdchsup else 0.0)]
            ,'XIXMEDCD':[float("Nan") if current_case_payments_none else (1.0 if current_case_xixmedcd else 0.0)]
            ,'SSIOTHER':[float("Nan") if current_case_payments_none else (1.0 if current_case_ssiother else 0.0)]
            ,'NOA':[float("Nan") if current_case_payments_none else (1.0 if current_case_noa else 0.0)]
            ,'FCMNTPAY'  :[float("Nan") if current_case_payments_none else float(current_case_fcmntpay)]
            ,'CLINDIS':[1.0 if child_clindis == 'Yes' else 2.0]
            ,'MR':[1.0 if child_mr_flag else 0.0]
            ,'VISHEAR':[1.0 if child_vishear_flag else 0.0]
            ,'PHYDIS':[1.0 if child_phydis_flag else 0.0]
            ,'EMOTDIST':[1.0 if child_emotdist_flag else 0.0]
            ,'OTHERMED':[1.0 if child_othermed_flag else 0.0]
            ,'CASEGOAL_1':[1.0 if current_case_goal == 'Reunification' else 0.0]
            ,'CASEGOAL_2':[1.0 if current_case_goal == 'Live with Other Relatives' else 0.0]
            ,'CASEGOAL_3':[1.0 if current_case_goal == 'Adoption' else 0.0]
            ,'CASEGOAL_4':[1.0 if current_case_goal == 'Long Term Foster Care' else 0.0]
            ,'CASEGOAL_5':[1.0 if current_case_goal == 'Emancipation' else 0.0]
            ,'CASEGOAL_6':[1.0 if current_case_goal == 'Guardianship' else 0.0]
            ,'CASEGOAL_7':[1.0 if current_case_goal == 'Goal Not Yet Established' else 0.0]
            ,'CASEGOAL_99':[0.0]
            ,'ISWAITING':[1.0 if child_iswaiting == 'Yes' else 0.0]
            ,'EVERADPT_1.0':[1.0 if child_everadpt == 'Yes' else 0.0]
            ,'EVERADPT_2.0':[1.0 if child_everadpt == 'No' else 0.0]
            #     #,'EVERADPT_3.0'
            ,'AGEADOPT_0.0':[1.0 if child_everadpt != 'Yes' else 0.0]
            ,'AGEADOPT_1.0':[1.0 if child_everadpt_age <= 2 else 0.0]
            ,'AGEADOPT_2.0':[1.0 if 2 < child_everadpt_age <= 5 else 0.0]
            ,'AGEADOPT_3.0':[1.0 if 5 < child_everadpt_age <= 12 else 0.0]
            ,'AGEADOPT_4.0':[1.0 if 12 < child_everadpt_age else 0.0]
            #     #,'AGEADOPT_5.0'
            #     #,'AGEADOPT_nan'
            ,'CTKFAMST_1.0':[1.0 if child_ctkfamst == 'Married Couple' else 0.0]
            ,'CTKFAMST_2.0':[1.0 if child_ctkfamst == 'Unmarried Couple' else 0.0]
            ,'CTKFAMST_3.0':[1.0 if child_ctkfamst == 'Single Female' else 0.0]
            ,'CTKFAMST_4.0':[1.0 if child_ctkfamst == 'Single Male' else 0.0]
            ,'CARETAKER_AGE':[float(child_caretakerage)]
            }
            #st.write(child_input_record_data)
            # Create child record input dataframe
            child_input_record_df = pd.DataFrame(child_input_record_data)


            ### RUN RECOMMENDER MODEL ###
            #regroup relevant user input for recommender model
            input_age = FMModel.regroup_age(child_birthday)
            input_race = FMModel.regroup_race(child_race, child_hispanic)
            input_placement = FMModel.regroup_placement(num_prev_placements)
            input_disability = FMModel.regroup_disability(child_clindis, child_mr_flag, child_vishear_flag, child_phydis_flag, child_emotdist_flag, child_othermed_flag)
            input_gender = FMModel.regroup_gender(child_gender)

            #loading configuration and datasets
            device, templatechilddf, ratingsdf, agelookupdf, racelookupdf, disabilitylookupdf, placementlookupdf, genderlookupdf, lenmodel, lenfeatures = FMModel.load_and_prep_datasets()

            #loading the model
            modelinfer = FMModel.load_model(lenmodel = lenmodel, lenfeatures = lenfeatures, device = device)

            #load providers 
            providers, provider_biases, provider_embeddings = FMModel.load_providers(ratingsdf = ratingsdf, modelinfer = modelinfer, device = device)

            #get user parameters from UI 
            childid, ageid,raceid,disability,placement,gender = FMModel.get_lookups(templatechilddf = templatechilddf, agelookupdf = agelookupdf, racelookupdf = racelookupdf, disabilitylookupdf = disabilitylookupdf, placementlookupdf = placementlookupdf, genderlookupdf = genderlookupdf, age = input_age,race = input_race, disability = input_disability, placement = input_placement, gender = input_gender)
            #st.write(childid, ageid,raceid,disability,placement,gender)
            
            #store output into variable
            recommender_output = FMModel.get_recommendations(modelinfer = modelinfer, device = device, providers = providers, provider_biases = provider_biases, provider_embeddings = provider_embeddings, childid = childid, raceid = raceid, ageid = ageid, disability = disability, placement = placement, gender = gender, topN = 12)
            #st.write(recommender_output)
            ### FINISH RUNNING RECOMMENDER MODEL ###


            ### SET UP DURATION MODEL ###
            providers_lookup = DurationModel.load_provider_lookup_table()
            recommended_providers = recommender_output.merge(providers_lookup, how = 'left', left_on = 'PROVIDER_ID', right_on = 'PROVIDER_ID')
            recommended_providers_features = recommended_providers[DurationModel.FOSTER_FEATURES].reset_index(drop=True)
            child_input_features = pd.concat([child_input_record_df]*recommended_providers_features.shape[0], ignore_index = True)
            placements_to_predict = pd.concat([child_input_features, recommended_providers_features], axis =1)
            #st.write(placements_to_predict)
            ### FINISH SET UP OF DURATION MODEL ###


            ### RUN DURATION AND PROBABILITY MODELS ###
            duration_error_table = DurationModel.load_duration_error_table()
            duration_model = DurationModel.load_duration_model()
            probability_model = DurationModel.load_positive_probability_model()
            duration_prediction = DurationModel.get_duration(duration_model, duration_error_table, placements_to_predict)
            probability_prediction = DurationModel.get_probability_of_good_outcome(probability_model, placements_to_predict)
            final_providers = pd.concat([recommended_providers, duration_prediction, probability_prediction], axis = 1)
            # st.write(final_providers)
            ### FINISH RUNNING DURATION AND PROBABILITY MODELS ###


            ### FORMAT OUTPUT ###
            # st.write(recommended_providers)
            # st.write(duration_prediction)
            st.text('')
            st.text('')
            st.text('')
            st.text('')
            st.title('Top Matched Providers')
            providers = st.beta_container()
            
            with providers:
                provcols =  st.beta_columns(3)
                button_dict1 = {}
                button_dict2 = {}
                button_dict3 = {}
                for index, row in final_providers.iterrows():
                    mods = index%3
                    if mods == 0:
                        with provcols[0]:
                            html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            html = str(index + 1) + ". " + html1
                            html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>' 
                            st.write(HTML_WRAPPER1.format(html), unsafe_allow_html=True)
                            button_dict2["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index))
                            if button_dict2["string{}".format(index)]:
                                DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            st.markdown("---")
                            
                    elif  mods == 1:
                        with provcols[1]:
                            html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            html = str(index + 1) + ". " + html1
                            html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>'
                            st.write(HTML_WRAPPER2.format(html), unsafe_allow_html=True)
                            button_dict2["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index))
                            if button_dict2["string{}".format(index)]:
                                DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            st.markdown("---")
                            
                    elif mods == 2: 
                        with provcols[2]:
                            html1 = "Unknown" if type(row["PROVIDER_NAME"])==float else '<em> <b>'+ row["PROVIDER_NAME"] + '</em> </b>'+ '    (Provider ID: ' + str(row["PROVIDER_ID"]) + ')'
                            html = str(index + 1) + ". " + html1
                            html1 = "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else str(round(row["PROVIDER_NUM_PREV_PLACEMENTS"]))
                            #html = html + '<br>' + 'Number of Children Fostered: ' + str(int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
                            html = html + '<br>' + 'Number of Children Fostered: ' + html1
                            html = html + '<br>' + "Provider Strengths: " + '<b>' + row["FLAGS"] + '</b>'
                            html1 = "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)
                            html2 = html + '<br>' + "Track Record for reunification/adoption/guardianship: " + str(html1) + '%'
                            html = html2 + '<br>' + "Match Rating: " + str(round(row.RATING,2)) + "/5"
                            html = html + '<br>'+ "Estimated Stay Duration: " + '<b>' + str(int(round(row["Predicted Duration"],0))) + '</b>' + " days"
                            html = html + '<br>'+ "Probability of Positive Outcome: " + '<b>' + str(round(row["Probability of Good Outcome"]*100,2)) + "%" + '</b>'
                            st.write(HTML_WRAPPER3.format(html), unsafe_allow_html=True)
                            button_dict2["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index))
                            if button_dict2["string{}".format(index)]:
                                DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
                            st.markdown("---")
                        
#                button_dict = {}
#                for index, row in final_providers.iterrows():
#                    st.write(str(index + 1),". ", "Unknown" if type(row["PROVIDER_NAME"])==float else row["PROVIDER_NAME"], '    (Provider ID: ', row["PROVIDER_ID"], ") ------- ", row["FLAGS"])
#                    # st.write("Flags: ", row["FLAGS"])
#                    st.write("Number of Children Fostered: ", "Unknown" if np.isnan(row["PROVIDER_NUM_PREV_PLACEMENTS"]) else int(round(row["PROVIDER_NUM_PREV_PLACEMENTS"])))
#                    st.write("Provider Strengths: ", "No Red Flags" if row["FLAGS"] == 'No Flags' else row["FLAGS"])
#                    st.write("Track Record for reunification/adoption/guardianship: ", "Unknown" if np.isnan(round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1)) else round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1), '%')
#                    st.write("Match Rating: ", round(row.RATING,2), "/5")
#                    st.write("Estimated Stay Duration: ", int(round(row["Predicted Duration"],0)), "days")
#                    st.write("Probability of Positive Outcome: ", round(row["Probability of Good Outcome"]*100,2), "%")
#                    #button_dict["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index),on_click =  DurationModel.get_probability_distribution, args = (placements_to_predict.iloc[[index]], probability_model))
#                    button_dict["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index))
#                    if button_dict["string{}".format(index)]:
#                        DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
#                    st.text('')
#                    st.text('')

    return None
    
def cs_home():
	st.write(BANNER,unsafe_allow_html=True) 
	st.write(BANNERsmall,unsafe_allow_html=True) 

	st.session_state['resetter'] = False
	image = Image.open('homepage_children.jpeg')
	st.image(image, width = 1200)
	
	st.title('Foster Care Matcher')
	st.write('The **foster care system** in the US is currently responsible for the lives and placements of over 500,000 children across the entire country today. Finding a good home for a foster child to stay in, while their biological parents take the time they need to recover from certain issues, can be a very difficult task. Foster parents (foster providers) are not always able to take care of a child and their specific needs. When this happens, a placement is disrupted, either by the foster provider requesting a change, or due to other complications such as the child running away or being admitted to an institution. Children who have to experience more instability like this have a higher chance of suffering from long term effects such as trauma or entering the Justice system. \n \n **Our team is focused on improving the foster child to foster provider matching process.** The system currently relies heavily on the expertise of specific foster placement specialists, without formally leveraging the insights available from historical placement information. Using merged data sources from the Adoption and Foster Care Analysis and Reporting dataset (AFCARS) - __annual case-level information of each child record in the foster care system mandated by the federal government__, and the Florida Removal and Placement History dataset (FRPH) - __granular data of each child’s placement details with extra information on duration__, we’ve built the **Foster Care Matcher**. \n \n **Foster Care Matcher** provides a list of top-quality foster care providers (parents) by utilizing a **Recommender System** powered by factorization machines that incorporates content and knowledge based, collaborative and contextual filtering with a customized match rating and model scoring configuration. To complement our Recommender System, a **Placement Duration Model** and an **Outcome Probability Model** will predict how long the current placement in question will last and what the probability of a good placement outcome will be to further assist a placement specialist in making a decision when trying to place a child.')

### JOURNEY PAGE ###
def cs_journey():
	st.write(BANNER,unsafe_allow_html=True) 
	st.write(BANNERsmall,unsafe_allow_html=True) 
	st.session_state['resetter'] = False
	header = st.beta_container()
	product = st.beta_container()

	with header:
		
		# Creating the Titles and Image	
		st.header("Child's Previous Placement Tracker")
		st.write("See all the previous placements for a specific Child, their previous foster providers, locations, and duration of placements.")
		st.subheader("Please select the Child ID")

	with product:	
		## initialize values
		# placed_before = 'Select one'
		child_ID = 'Select one'

		#load data
		
		df = pd.read_csv('child_demo.csv')
		@st.cache
		def dataload(df, cid = child_ID):
			source = df[df.AFCARS_ID==cid]
			source['zip'] = source['zip'].astype('str')
			source['END_REASON'] = source['END_REASON'].astype('str')
			source['REMOVAL_RANK'] = source['REMOVAL_RANK'].astype('int')
			source['PLACEMENT_BEGIN_DATE'] = source['PLACEMENT_BEGIN_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
			source['PLACEMENT_END_DATE'] = source['PLACEMENT_END_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
			source['REMOVAL_DATE'] = source['REMOVAL_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    		
			return source
			
		def plot_multi(source):
			# import geopandas as gpd
			gdf = gpd.read_file('https://raw.githubusercontent.com/python-visualization/folium/master/tests/us-states.json', driver='GeoJSON')
			gdf = gdf[gdf.id=='FL']
			base = alt.Chart(gdf).mark_geoshape(
			stroke='gray', 
			fill='lightgrey')

			points = alt.Chart(source).mark_circle().encode(
			longitude='longitude:Q',
			latitude='latitude:Q',
			color = alt.value('steelblue'),
			size=alt.Size('PLACEMENT_LENGTH:Q', title='Placement Length'),
			# title='placement locaton in Florida',
			tooltip=['REMOVAL_RANK:Q','PROVIDER_ID:Q','END_REASON:Q']
			).properties(title='Placement Location')
			
			#g_plot = base + points
			# st.write(g_plot)

			pl_num_mark = alt.Chart(source).mark_circle().encode(
			x='PLACEMENT_NUM',
			y='PLACEMENT_LENGTH:Q',
			size='PLACEMENT_LENGTH',
			color = 'zip',
			tooltip=['REMOVAL_RANK','PROVIDER_ID','PLACEMENT_NUM', 'PLACEMENT_LENGTH','END_REASON']
			).properties(title='Placement Length vs Number').interactive()
			
			pl_duration_mark = alt.Chart(source).mark_circle().encode(
			# x='PLACEMENT_NUM:Q',
			x='PLACEMENT_BEGIN_DATE:T',
			y= 'PLACEMENT_LENGTH:Q',
			size='PLACEMENT_LENGTH',
			tooltip=['REMOVAL_RANK','PROVIDER_ID','PLACEMENT_BEGIN_DATE', 'PLACEMENT_LENGTH','END_REASON']
			).properties(
				title='Placement Duration').interactive()

			# plot_group1 = alt.hconcat(pl_duration_mark, pl_num_mark, points)
			plot_group1 = alt.hconcat(pl_duration_mark, pl_num_mark) 
    
			return plot_group1

		# placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

		# if placed_before == 'Yes':
		child_ID = st.selectbox('Child ID is a unique idenfier of each child in foster care system', 
		[80000001,451000749,1811000629,8291010319,261405401,81010219,251010479,1010299,31010759])

	
		source = dataload(df, cid = child_ID)
		pl_num = source.shape[0]
		pl_start = source['PLACEMENT_BEGIN_DATE'].min()
		pl_yrs =str(source['PLACEMENT_END_DATE'].max().year - source['PLACEMENT_BEGIN_DATE'].min().year)
		
		st.subheader('Map showing Child\'s previous placement locations')
		
		plot_group1 = plot_multi(source)
				
		df_map = source[['PLACEMENT_LENGTH','latitude', 'longitude']]
		st.map(df_map)	
		
		st.write('This child has experienced ' + str(pl_num) + ' placements within ' + str(pl_yrs) + ' years with mixed experience.')
		
		st.subheader('Graphs for Duration and Placement Order Analysis')
		st.write(" ")
		

		st.write(plot_group1)

def cs_architecture():
    st.write(BANNER,unsafe_allow_html=True) 
    st.write(BANNERsmall,unsafe_allow_html=True) 

    st.session_state['resetter'] = False
    st.title('E2E Pipilines and Models Specifications')
    st.text("")
    product1 = st.beta_container()
    product2, product3 =  st.beta_columns(2)
    product4, product5 = st.beta_columns(2)

    with product1:
        st.header('ML Pipelines and App Deployment')
        image = Image.open('APPGCPdiagram.png').convert('RGB').save('pipeline_mk_new.png')
        image = Image.open('pipeline_mk_new.png')   
        st.image(image, width = 1100)

    with product2:
        st.header('Dive Into Our Match Recommender System')
        image2 = Image.open('FM_ModelSpecification.png').convert('RGB').save('recommender_new.png')
        image2 = Image.open('recommender_new.png')
        st.image(image2, width = 500)

    with product3:
        st.header('Similarity Analysis and Feature Selection')
        image3 = Image.open('FeatureSelectionRandomForest.png').convert('RGB').save('FeatureSelection_new.png')
        image3 = Image.open('FeatureSelection_new.png')
        st.image(image3, width = 600)

    with product4:
        st.header('Placement Duration Model Evaluation')
        image4 = Image.open('duration_evaluation.png')
        st.image(image4, width = 550)

    with product5:
        st.header('Outcome Model Evaluation')
        image5 = Image.open('outcome_evaluation.png')
        st.image(image5, width = 550)



# def cs_model():
# 	st.write(BANNER,unsafe_allow_html=True) 
# 	st.write(BANNERsmall,unsafe_allow_html=True) 

# 	st.session_state['resetter'] = False
# 	st.title('Foster Care Matcher')
# 	st.header('Features about Foster Care Matcher')
# 	st.write('Process on creating this')
# 	model2 = XGBRegressor(objective ='reg:tweedie', tree_method = "gpu_hist", max_depth=12, n_estimators=200, predictor='cpu_predictor')
# 	model2.load_model("./XGBoost_regressor_2")
# 	placements_to_predict = pd.read_csv("./placements_to_predict.csv")
# 	st.write(placements_to_predict)
# 	new_df = model2.predict(placements_to_predict)
# 	st.write(new_df)

def cs_team():
    st.write(BANNERleft,unsafe_allow_html=True) 
    st.write(BANNERleftsmall,unsafe_allow_html=True) 

    st.session_state['resetter'] = False
    st.title('Our Team')
    st.text("")
    st.text("")
    picture_jason = Image.open('picture_jason.jpg')
    picture_james = Image.open('picture_james.jpg')
    picture_vineetha = Image.open('picture_vineetha.jpg')
    picture_christina = Image.open('picture_christina.jpg')


    col1, col2, col3 = st.beta_columns(3)

    col1.image(picture_jason, width = 300)
    col1.write('<div style="text-align: left"> <b> Jason Papale </b> </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> MIDS Class of 2021 </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> <b> University of California, Berkeley </b> </div>', unsafe_allow_html = True)
    col1.text("")
    col1.text("")
    col1.text("")
    

    col1.image(picture_james, width = 300)
    col1.write('<div style="text-align: left"> <b> James Gao </b> </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> MIDS Class of 2021 </div>', unsafe_allow_html = True)
    col1.write('<div style="text-align: left"> <b> University of California, Berkeley </b> </div>', unsafe_allow_html = True)
    col1.text("")
    col1.text("")
    col1.text("")
    col1.text("")


    col2.image(picture_christina, width = 300)
    col2.write('<div style="text-align: left"> <b> Christina Min </b> </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> MIDS Class of 2021 </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> <b> University of California, Berkeley </b> </div>', unsafe_allow_html = True)
    col2.text("")
    col2.text("")
    col2.text("")
    

    col2.image(picture_vineetha, width = 300)
    col2.write('<div style="text-align: left"> <b> Vineetha Nalini </b> </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> MIDS Class of 2021 </div>', unsafe_allow_html = True)
    col2.write('<div style="text-align: left"> <b> University of California, Berkeley </b> </div>', unsafe_allow_html = True)
    col2.text("")
    col2.text("")
    col2.text("")
    col2.text("")



    st.title("Acknowledgments")
    st.write("Thank you for all of your support for this project!")
    st.text("")
    st.write('**Joyce Shen** - UC Berkeley School of Information, Lecturer')
    st.write('**David Steier** - UC Berkeley & Carnegie Mellon University, Professor')
    st.write('**Robert Latham** - University of Miami Law School, Associate Director')
    st.write('**Roshannon Jackson** - State of Florida, Leon County, Director of Placements')
    st.write('**Colorado Reed** - UC Berkeley School of Information, Teaching Assistant and PhD')


# Run main()

if __name__ == '__main__':
    main()
