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
# import geopandas as gpd
from PIL import Image
import time

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")

#### Creating pages for website
st.sidebar.title('Foster Care Matcher')
mypage = st.sidebar.radio('Pages', ['Home', 'Matcher', 'Journey', 'Architecture', 'Modeling', 'Team'])

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



### HOME PAGE ###
if mypage == 'Home':
	image = Image.open('homepage_children.jpg')
	st.image(image, width = 1200)
	
	st.title('Foster Care Matcher')
	st.markdown("""
	### This app provides a list of top quality matched providers to a prospect foster child with expected placement duration and probability of placement""")
	st.write('Team **Foster Care Matching** or **CS: FB** (*A child saved, a future brightened*) or **Forever Foster** or **Foster Forever(F2)** *Love.Heal.Trust.Respect.Cherish* is focused on improving the current foster care matching system which heavily relies on domain expertise and specific requests from foster parents that may hinder the potential of leveraging the insights from all the historic placement information of paired foster parents and children. This matching task could affect **200,000** children in Florida alone and **500,000** in US.  \n  \nUsing merged data sources from Adoption and Foster Care Analysis and Reporting (***AFCARS***) - annual case-level information of each child record in foster care system mandated by federal government; Florida Removal and Placement History (***FRPH***) - granular data of each child placement details with extra information on duration, date of start and end of placement; Demographics of Children including race, gender, date of birth, etc.  \n  \nWe have built a **Foster Care Matching Recommender System** by providing top **5** to **20** top-quality matched providers for each child entering the system using cutting-edge *factorization machines* that incorporates content-based, knowledge-based, collaborative filtering and contextual filtering with our customized match rating and model scoring configuration.  \n  \nTo complement our recommender system, we also created a **Placement Duration Model** and **Outcome Probability Model** that will predict how long the placement will last and what is the probability of a good placement outcome for our MVP to foster care placement specialists.  \n  \nWe intend to launch our application to foster care placement specialists by Aug 3rd.')



### DEMO PAGE ###
elif mypage == 'Matcher':
	header = st.beta_container()
	product = st.beta_container()

	with header:
		# Creating the Titles and Image	
		st.title("Foster Care Matcher")
		st.header("Find the right foster care provider for your child")
		st.subheader("Use all of the existing available data on previous placements to find the best Provider whose suited to care for the new foster child")


	with product:
		## initialize values
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
		# child_multi_dis = 'Select all that apply'
		child_everadpt = 'Select one'
		child_everadpt_age = float("Nan")
		current_case_goal = 'Select one'
		find_providers_button = None
		## need to add this so session state if resetter in session state blah blah blah
		if 'resetter' not in st.session_state:
			st.session_state['resetter'] = False

		row0_1, row0_2 = st.beta_columns(2)
		with row0_1:
			placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])
		with row0_2:
			if placed_before == 'Yes':
				num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

		if num_prev_placements > 0:
			st.header("Previous Placement Information")
			row1_1, row1_2 = st.beta_columns(2)
			with row1_1:
				child_date_of_first_placement = st.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			with row1_2:
				child_num_prev_placements_good = st.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			row2_1, row2_2 = st.beta_columns(2)
			with row2_1:
				child_num_prev_placements_bad = st.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements-child_num_prev_placements_good, step = 1)
			with row2_2:
				# st.text("Remaining placements will be counted as having a NEUTRAL outcome.")
			# st.text("")
				child_recent_placement_outcome = st.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])

		if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
			st.header("Child and Removal Information")
			row3_1, row3_2 = st.beta_columns(2)
			with row3_1:
				child_ctkfamst = st.selectbox("What kind of caretaker was the child removed from?", ['Select one', 'Married Couple', 'Unmarried Couple', 'Single Female', 'Single Male'])
			with row3_2:
				child_caretakerage = st.number_input("Enter the age (in years) of the child's caretaker at the time of removal? (If there was more than one caretaker, put the age of just one of them)", min_value = 0, max_value = 100, step = 1)
			row4_1, row4_2 = st.beta_columns(2)
			with row4_1:
				child_birthday = st.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			with row4_2:
				child_gender = st.selectbox("Child's gender", ['Select one', 'Male', 'Female'])
			row5_1, row5_2 = st.beta_columns(2)
			with row5_1:
				child_race = st.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
			with row5_2:
				child_hispanic = st.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'])

		if child_hispanic != 'Select one':
			st.text("")
			st.header("Child's Disabilities")
			row6_1, row6_2 = st.beta_columns(2)
			with row6_1:
				child_clindis = st.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])

		if child_clindis == 'Yes':
			with row6_2:
				st.write("Check all that apply:")
				# child_multi_dis = st.multiselect('Select all that apply', ["Mental Retardation","Visually or Hearing Impaired","Physically Disabled","Emotionally Disturbed","Other Medically Diagnosed Condition"])
				child_mr_flag = st.checkbox("Mental Retardation")
				child_vishear_flag = st.checkbox("Visually or Hearing Impaired")
				child_phydis_flag = st.checkbox("Physically Disabled")
				child_emotdist_flag = st.checkbox("Emotionally Disturbed")
				child_othermed_flag = st.checkbox("Other Medically Diagnosed Condition")

		if child_clindis != 'Select one':
			st.text("")
			row7_1, row7_2 = st.beta_columns(2)
			with row7_1:
				child_iswaiting = st.selectbox("Is the child currently waiting for adoption?", ['Select one', 'Yes', 'No'])
			with row7_2:
				child_everadpt = st.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'])

		if child_everadpt == 'Yes':
			row8_1, row8_2 = st.beta_columns(2)
			with row8_1:
				child_everadpt_age = st.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18)
			with row8_2:
				st.text("")

		if child_everadpt != 'Select one':
			st.text("")
			st.subheader("Why did the child enter the foster care system? (Check all that apply)")
			
			row9_1, row9_2, row9_3 = st.beta_columns(3)
			with row9_1:
				physical_abuse = st.checkbox('Physical Abuse')
				sexual_abuse = st.checkbox('Sexual Abuse')
				emotional_abuse_neglect = st.checkbox('Emotional Abuse')
				physical_neglect = st.checkbox("Physical Neglect")
				medical_neglect = st.checkbox("Medical Neglect")
				alcohol_abuse_child = st.checkbox("Child's Alcohol Abuse")
				drug_abuse_child = st.checkbox("Child's Drug Abuse")
				child_behavior_problem = st.checkbox('Child Behavior Problem')
				child_disability = st.checkbox('Child Disability')
				transition_to_independence = st.checkbox("Transition to Independence")
			with row9_2:
				inadequate_supervision = st.checkbox("Inadequate Supervision")
				adoption_dissolution = st.checkbox("Adoption Dissolution")
				abandonment = st.checkbox("Abandonment")
				labor_trafficking = row9_2.checkbox("Labor Trafficking")
				sexual_abuse_sexual_exploitation = st.checkbox("Sexual Exploitation")
				prospective_physical_abuse = st.checkbox("Prospective Physical Abuse")
				prospective_sexual_abuse = st.checkbox('Prospective Sexual Abuse')
				prospective_emotional_abuse_neglect = st.checkbox("Prospective Emotional Abuse")
				prospective_physical_neglect = st.checkbox('Prospective Physical Neglect')
				prospective_medical_neglect = st.checkbox("Prospective Medical Neglect")
			with row9_3:
				alcohol_abuse_parent = st.checkbox("Parent's Alcohol Abuse")
				drug_abuse_parent = st.checkbox("Parent's Drug Abuse")
				incarceration_of_parent = st.checkbox('Incarceration of Parent')
				death_of_parent = st.checkbox('Death of Parent')
				domestic_violence = st.checkbox("Domestic Violence")
				inadequate_housing = st.checkbox("Inadequate Housing")
				caregiver_inability_to_cope = st.checkbox("Caregiver's inability to cope")
				relinquishment = st.checkbox('Relinquishment')
				request_for_service = st.checkbox('Request for Service')
				csec = st.checkbox("CSEC")

			st.header("Current placement information")
			row10_1, row10_2, row10_3 = st.beta_columns(3)
			with row10_1:
				current_case_goal = st.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
			
		if current_case_goal != 'Select one':
			st.text("")
			with row10_2:
				st.write("Current placement's applicable payments")
				current_case_ivefc = st.checkbox("Foster Care Payments")
				current_case_iveaa = st.checkbox("Adoption Assistance")
				current_case_ivaafdc = st.checkbox("TANF Payment (Temporary Assistance for Needy Families)")
				current_case_ivdchsup = st.checkbox("Child Support Funds")
				current_case_xixmedcd = st.checkbox("Medicaid")
				current_case_ssiother = st.checkbox("SSI or Social Security Benefits")
				current_case_noa = st.checkbox("Only State or Other Support")
				current_case_payments_none = st.checkbox("None of the above apply")
			with row10_3:
				current_case_fcmntpay = st.number_input("Monthly Foster Care Payment ($)", min_value = 0, step = 100)
			st.text("")
			st.text("")
			find_providers_button = st.button("Find Providers")


		## Once the button is pressed, the resetter will be set to True and will be updated in the Session State
		if find_providers_button:
			st.session_state['resetter'] = True

			# st.balloons()
		## Recommender System output 
		if st.session_state['resetter'] == True:
			

			## construct child record using user_input
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
			# 	#,REMOVAL_LENGTH #Need to make reflective as of placement begin date   
			# 	#,PLACEMENT_NUMBER #Need to apply after using James's flattened version
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
			# 	#,'MOVE_MILES'
			# 	#,'ROOMMATE_COUNT'
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
			# 	#,'EVERADPT_3.0'
			,'AGEADOPT_0.0':[1.0 if child_everadpt != 'Yes' else 0.0]
			,'AGEADOPT_1.0':[1.0 if child_everadpt_age <= 2 else 0.0]
			,'AGEADOPT_2.0':[1.0 if 2 < child_everadpt_age <= 5 else 0.0]
			,'AGEADOPT_3.0':[1.0 if 5 < child_everadpt_age <= 12 else 0.0]
			,'AGEADOPT_4.0':[1.0 if 12 < child_everadpt_age else 0.0]
			# 	#,'AGEADOPT_5.0'
			# 	#,'AGEADOPT_nan'
			,'CTKFAMST_1.0':[1.0 if child_ctkfamst == 'Married Couple' else 0.0]
			,'CTKFAMST_2.0':[1.0 if child_ctkfamst == 'Unmarried Couple' else 0.0]
			,'CTKFAMST_3.0':[1.0 if child_ctkfamst == 'Single Female' else 0.0]
			,'CTKFAMST_4.0':[1.0 if child_ctkfamst == 'Single Male' else 0.0]
			,'CARETAKER_AGE':[float(child_caretakerage)]
			}

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
			device, ratingsdf, agelookupdf, racelookupdf, disabilitylookupdf, placementlookupdf, genderlookupdf, lenmodel, lenfeatures = FMModel.load_and_prep_datasets()

			#loading the model
			modelinfer = FMModel.load_model(lenmodel = lenmodel, lenfeatures = lenfeatures, device = device)

			#load providers 
			providers, provider_biases, provider_embeddings = FMModel.load_providers(ratingsdf = ratingsdf, modelinfer = modelinfer, device = device)

			#get user parameters from UI 
			ageid,raceid,disability,placement,gender = FMModel.get_lookups(agelookupdf = agelookupdf, racelookupdf = racelookupdf, disabilitylookupdf = disabilitylookupdf, placementlookupdf = placementlookupdf, genderlookupdf = genderlookupdf, age = input_age,race = input_race, disability = input_disability, placement = input_placement, gender = input_gender)

			#store output into variable
			recommender_output = FMModel.get_recommendations(modelinfer = modelinfer, device = device, providers = providers, provider_biases = provider_biases, provider_embeddings = provider_embeddings, raceid = raceid, ageid = ageid, disability = disability, placement = placement, gender = gender, topN = 12)
			### FINISH RUNNING RECOMMENDER MODEL ###


			### SET UP DURATION MODEL ###
			providers_lookup = DurationModel.load_provider_lookup_table()
			recommended_providers = recommender_output.merge(providers_lookup, how = 'left', left_on = 'PROVIDER_ID', right_on = 'PROVIDER_ID')
			recommended_providers_features = recommended_providers[DurationModel.FOSTER_FEATURES].reset_index(drop=True)
			child_input_features = pd.concat([child_input_record_df]*recommended_providers_features.shape[0], ignore_index = True)
			placements_to_predict = pd.concat([child_input_features, recommended_providers_features], axis =1)
			### FINISH SET UP OF DURATION MODEL ###


			### RUN DURATION AND PROBABILITY MODELS ###
			duration_error_table = DurationModel.load_duration_error_table()
			duration_model = DurationModel.load_duration_model()
			probability_model = DurationModel.load_positive_probability_model()
			duration_prediction = DurationModel.get_duration(duration_model, duration_error_table, placements_to_predict)
			probability_prediction = DurationModel.get_probability_of_good_outcome(probability_model, placements_to_predict)
			final_providers = pd.concat([recommended_providers, duration_prediction, probability_prediction], axis = 1)
			### FINISH RUNNING DURATION AND PROBABILITY MODELS ###

			# st.balloons()
			### FORMAT OUTPUT ###
			# st.write(recommended_providers)
			# st.write(duration_prediction)
			st.text('')
			st.text('')
			st.text('')
			st.text('')
			st.title('Top Matched Providers')
			button_dict = {}
			for index, row in final_providers.iterrows():
				st.write(str(index + 1),". ", row["PROVIDER_NAME"], '    (Provider ID: ', row["PROVIDER_ID"], ") ------- ", row["FLAGS"])
				# st.write("Flags: ", row["FLAGS"])
				st.write("Number of Children Fostered: ", row["PROVIDER_NUM_PREV_PLACEMENTS"])
				st.write("Positive Placement Outcome rate: ", round(row["PROVIDER_NUM_PREV_PLACEMENTS_GOOD_PERC"]*100,1), '%')
				st.write("Match Rating: ", round(row.RATING,2))
				st.write("Estimated Stay Duration: ", int(round(row["Predicted Duration"],0)), "days")
				st.write("Probability of Positive Outcome: ", round(row["Probability of Good Outcome"]*100,2), "%")
				button_dict["string{}".format(index)] = st.button("See Breakdown of Outcome Predictions", key = str(index))
				if button_dict["string{}".format(index)]:
					DurationModel.get_probability_distribution(placements_to_predict.iloc[[index]], probability_model)
				st.text('')
				st.text('')

			# st.balloons()

### JOURNEY PAGE ###
elif mypage == 'Journey':
	header = st.beta_container()
	product = st.beta_container()

	with header:
		# Creating the Titles and Image	
		st.header("My Journey with Foster Care")
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
			source['PLACEMENT_BEGIN_DATE'] = source['PLACEMENT_BEGIN_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
			source['PLACEMENT_END_DATE'] = source['PLACEMENT_END_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
			source['REMOVAL_DATE'] = source['REMOVAL_DATE'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    		
			return source
			
		def plot_multi(source):
			# import geopandas as gpd
			# gdf = gpd.read_file('https://raw.githubusercontent.com/python-visualization/folium/master/tests/us-states.json', driver='GeoJSON')
			# gdf = gdf[gdf.id=='FL']
			# base = alt.Chart(gdf).mark_geoshape(
			# stroke='gray', 
			# fill='lightgrey')	

			points = alt.Chart(source).mark_circle().encode(
			longitude='longitude:Q',
			latitude='latitude:Q',
			color = 'zip:N',
			size='PLACEMENT_LENGTH',
			# title='placement locaton in Florida',
			tooltip=['zip', 'PLACEMENT_LENGTH']
			).properties(title='placement location')

			# g_plot = base + points
			# st.write(g_plot)

			pl_num_mark = alt.Chart(source).mark_circle().encode(
			x='PLACEMENT_NUM',
			y='PLACEMENT_LENGTH:Q',
			size='PLACEMENT_LENGTH',
			color = 'zip',
			tooltip=['PLACEMENT_NUM', 'PLACEMENT_LENGTH']
			).properties(title='placement length vs number').interactive()
			
			pl_duration_mark = alt.Chart(source).mark_circle().encode(
			# x='PLACEMENT_NUM:Q',
			x='PLACEMENT_BEGIN_DATE:T',
			y= 'PLACEMENT_LENGTH:Q',
			size='PLACEMENT_LENGTH',
			tooltip=['PLACEMENT_BEGIN_DATE', 'PLACEMENT_LENGTH']
			).properties(
				title='placement duration').interactive()

			# plot_group1 = alt.hconcat(pl_duration_mark, pl_num_mark, points)
			plot_group1 = alt.hconcat(pl_duration_mark, pl_num_mark) 
    
			return plot_group1

		# placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

		# if placed_before == 'Yes':
		child_ID = st.selectbox('Child ID is a unique idenfier of each child in foster care system', 
		[80000001,451000749,1811000629,8291010319,261405401,81010219,251010479,1010299,31010759])

	
		# if num_prev_placements == 50:
		# st.markdown("""this is the journey of a child""")
		# st.subheader('this is the past journey of this child going through foster care system at least ' +str(num_prev_placements)+ ' times!')
		source = dataload(df, cid = child_ID)
		pl_num = source.shape[0]
		pl_start = source['PLACEMENT_BEGIN_DATE'].min()
		pl_yrs =str(source['PLACEMENT_END_DATE'].max().year - source['PLACEMENT_BEGIN_DATE'].min().year)

		st.write('this child has experienced ' + str(pl_num) + ' placements within ' + str(pl_yrs) + ' years with mixed experience.')
		# st.write(source.head(2))
		
		df_map = source[['PLACEMENT_LENGTH','latitude', 'longitude']]
		st.map(df_map)	

		plot_group1 = plot_multi(source)
		st.write(plot_group1)



### ARCHITECTURE PAGE ###
elif mypage == 'Architecture':
	st.title('Architecture and Workflow of Foster Care Matcher')
	# st.header('Features about Foster Care Matcher')
	# st.write('Process on creating this')
	# set the page layout
	# st.set_page_config(layout="wide")

	header = st.beta_container()
	# product = st.beta_container()

	with header:
		# st.title('Foster Care Matcher ')
		# st.header('Features about Foster Care Matcher')
		st.header('this is end to end pipeline from data ingestion to app deloyment')
		image = Image.open('Architecture_design2.png').convert('RGB').save('architecture_new.png')
		image = Image.open('architecture_new.png')
		st.image(image, width = 1000)
		# st.title('Foster Care Matcher')
		# st.header('Features about Foster Care Matcher')
		# st.subheader('this is data pipeline and ML pipeline')
		# st.write('Process on creating this')


	# with product:
	# 	st.subheader('this is the past journey of this child through foster care system ')
	# 	df = pd. DataFrame(
	# 	np.random.randn(200, 3),
	# 	columns=['a', 'b', 'c'])

	# 	c = alt.Chart(df).mark_circle().encode(
	# 		x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
	
	# 	st.write(c)

### MODELING PAGE ###
elif mypage == 'Modeling':
	st.title('Foster Care Matcher')
	st.header('Features about Foster Care Matcher')
	st.write('Process on creating this')
	model2 = XGBRegressor(objective ='reg:tweedie', tree_method = "gpu_hist", max_depth=12, n_estimators=200, predictor='cpu_predictor')
	model2.load_model("./XGBoost_regressor_2")
	placements_to_predict = pd.read_csv("./placements_to_predict.csv")
	st.write(placements_to_predict)
	new_df = model2.predict(placements_to_predict)
	st.write(new_df)



### TEAM PAGE ###
elif mypage == 'Team':
	st.title('Foster Care Matcher')
	st.header('Who worked on Information')
	st.write('Pictures')
	st.title("We'd like to thank")
	st.header('Orgs')
	st.write('Robert and Roshannon, David and Joyce')	















# import streamlit as st
# import pandas as pd
# import numpy as np
# import datetime as dt
# import pickle
# from xgboost import XGBClassifier, XGBRegressor


# st.write("start")

# new_button = st.button("click this")

# if new_button:
# 	model2 = XGBRegressor(objective ='reg:tweedie', tree_method = "gpu_hist", max_depth=12, n_estimators=200, predictor='cpu_predictor')
# 	model2.load_model("./XGBoost_regressor_2")
# 	placements_to_predict = pd.read_csv("./placements_to_predict.csv")
# 	st.write(placements_to_predict)
# 	new_df = model2.predict(placements_to_predict)
# 	st.write(new_df)










