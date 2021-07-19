import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import altair as alt
import geopandas as gpd
from PIL import Image

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")

## another small change

#### Creating pages for website
st.sidebar.title('Foster Care Matcher')

mypage = st.sidebar.radio('Pages', ['Home', 'Matcher', 'Matcher - Select Box', 'Matcher - Form Button', 'Architecture', 'Modeling', 'Team'])

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
	# image = Image.open('homepage_image.jpg').convert('RGB').save('fcare2.jpg')
	image = Image.open('homepage_image.jpg')
	st.image(image, width = 800)
	
	st.title('Foster Care Matcher')
	# st.header('Description about Foster Care Matcher')
	st.markdown("""
### This app provides a list of top quality matched providers to a prospect foster child with expected placement duration and probability of placement
""")
	st.write('Team **Foster Care Matching** or **CS: FB** (*A child saved, a future brightened*) or **Forever Foster** or **Foster Forever(F2)** *Love.Heal.Trust.Respect.Cherish* is focused on improving the current foster care matching system which heavily relies on domain expertise and specific requests from foster parents that may hinder the potential of leveraging the insights from all the historic placement information of paired foster parents and children. This matching task could affect **200,000** children in Florida alone and **500,000** in US.  \n  \nUsing merged data sources from Adoption and Foster Care Analysis and Reporting (***AFCARS***) - annual case-level information of each child record in foster care system mandated by federal government; Florida Removal and Placement History (***FRPH***) - granular data of each child placement details with extra information on duration, date of start and end of placement; Demographics of Children including race, gender, date of birth, etc.  \n  \nWe have built a **Foster Care Matching Recommender System** by providing top **5** to **20** top-quality matched providers for each child entering the system using cutting-edge *factorization machines* that incorporates content-based, knowledge-based, collaborative filtering and contextual filtering with our customized match rating and model scoring configuration.  \n  \nTo complement our recommender system, we also created a **Placement Duration Model** and **Outcome Probability Model** that will predict how long the placement will last and what is the probability of a good placement outcome for our MVP to foster care placement specialists.  \n  \nWe intend to launch our application to foster care placement specialists by Aug 3rd.')



### DEMO PAGE ###
elif mypage == 'Matcher':
	#---------------------------------#
	# Page layout (continued)
	## Divide page to 2 columns (col1 = questions' list, col2  = graph contents)
	# col1 = st.sidebar
	# col1, col2 = st.beta_columns((1,1))
	# col1.header = st.beta_container()
	header = st.beta_container()
	# col1.subheader = st.beta_container()
	product = st.beta_container()  
	# col1.product = st.beta_container()
	# add column for graphs	
	# graph = st.beta_columns()
	# col2.header = st.beta_container()
	# col2.graph = st.beta_container()

	# with col1.header:
	with header:
		# Creating the Titles and Image	
		st.title("Foster Care Matcher")
		st.header("Find the right foster care provider for your child")
		# st.write("To start, please answer foloowing questions.")
		st.subheader("To start, please answer foloowing questions.")

    # with col1.product:
	with product:	
		## initialize values
		placed_before = 'Select one'
		num_prev_placements = 0
		child_num_prev_placements_good = 0
		child_num_prev_placements_bad = 0
		child_date_of_first_placement = datetime.date(9999,1,1)
		child_recent_placement_outcome = 'Select one'
		child_hispanic = 'Select one'
		child_mr_flag = False
		child_vishear_flag = False
		child_phydis_flag = False
		child_emotdist_flag = False
		child_othermed_flag = False
		child_clindis = 'Select one'
		child_everadpt = 'Select one'
		current_case_goal = 'Select one'
		find_providers_button = None
		resetter = False

		#load data
		df = pd.read_csv('high_placement_children.csv')
		def dataload(df, pl_no = num_prev_placements):
			cid = df[df.PLACEMENT_NUM==pl_no]['AFCARS_ID'].unique()[0]
			source = df[df.AFCARS_ID==cid]
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
			color = 'zip:N',
			size='PLACEMENT_LENGTH',
			# title='placement locaton in Florida',
			tooltip=['zip', 'PLACEMENT_LENGTH']
			).properties(
				title='placement location')
			g_plot = base + points
			# st.write(g_plot)

			pl_number_line = alt.Chart(source).mark_line().encode(
			x='PLACEMENT_BEGIN_DATE:T',
			y= 'PLACEMENT_NUM:Q',
			tooltip=['PLACEMENT_BEGIN_DATE', 'PLACEMENT_NUM']
			).properties(
				title='placement journey'
			)
		
			pl_bar = alt.Chart(source).mark_bar().encode(
			x='PLACEMENT_NUM',
			y='PLACEMENT_LENGTH:Q',
			# title='placement locaton in Florida',
			tooltip=['PLACEMENT_NUM', 'PLACEMENT_LENGTH']
			).properties(
				title='placement duration')
			
			pl_duration_mark = alt.Chart(source).mark_circle().encode(
			x='PLACEMENT_BEGIN_DATE:T',
			y= 'PLACEMENT_LENGTH:Q',
			size='PLACEMENT_LENGTH',
			tooltip=['PLACEMENT_BEGIN_DATE', 'PLACEMENT_LENGTH']
			).properties(
				title='placement duration')

			plot_group1 = alt.hconcat(pl_number_line, pl_duration_mark, points) #, g_plot
			plot_group2 = alt.hconcat(pl_number_line, pl_bar, points)
    
			return plot_group1, plot_group2

		placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

		if placed_before == 'Yes':
			num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

		if (num_prev_placements > 0) & (num_prev_placements == 50):
			# if num_prev_placements == 50:
			# st.markdown("""this is the journey of a child""")
			st.subheader('this is the past journey of this child going through foster care system at least ' +str(num_prev_placements)+ ' times!')

			# df = pd.read_csv('high_placement_children.csv')
			# def dataload(df, pl_no = num_prev_placements):
			# 	cid = df[df.PLACEMENT_NUM==pl_no]['AFCARS_ID'].unique()[0]
			# 	source = df[df.AFCARS_ID==cid]
			# 	return source
			source = dataload(df, pl_no = num_prev_placements)
			pl_num = source['PLACEMENT_NUM'].max()
			pl_start = source['PLACEMENT_BEGIN_DATE'].min()
			# pl_yrs = str(source['PLACEMENT_END_DATE'].year.max() - source['PLACEMENT_BEGIN_DATE'].year.min())
			# st.write('this child has experienced ' + str(pl_num) + ' placements in ' + str(pl_yrs) + ' years since ' + str(pl_start))
			st.write('this child has experienced ' + str(pl_num) + ' placements since ' + str(pl_start))
			# st.write(source.head(2))

			plot_group1, plot_group2 = plot_multi(source)
			st.write(plot_group1)
			st.write(plot_group2)

			# st.map()
				# df = pd. DataFrame(
				# np.random.randn(200, 3),
				# columns=['a', 'b', 'c'])

				# c = alt.Chart(df).mark_circle().encode(
				# 	x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
			
				# st.write(c)
		if (num_prev_placements > 0) & (num_prev_placements < 50):	
			st.header("Previous Placement Information")
			child_date_of_first_placement = st.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_num_prev_placements_good = st.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			child_num_prev_placements_bad = st.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			st.text("Remaining placements will be counted as having a NEUTRAL outcome.")
			st.text("")
			child_recent_placement_outcome = st.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])

		if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
			st.header("Child Information")
			child_birthday = st.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_race = st.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
			child_hispanic = st.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'])

		if child_hispanic != 'Select one':
			st.text("")
			st.write("Child's Disabilities")
			child_clindis = st.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])

		if child_clindis == 'Yes':
			st.write("Check all that apply:")
			child_mr_flag = st.checkbox("Mental Retardation")
			child_vishear_flag = st.checkbox("Visually or Hearing Impaired")
			child_phydis_flag = st.checkbox("Physically Disabled")
			child_emotdist_flag = st.checkbox("Emotionally Disturbed")
			child_othermed_flag = st.checkbox("Other Medically Diagnosed Condition")

		if child_clindis != 'Select one':
			st.text("")
			child_everadpt = st.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'])


		if child_everadpt == 'Yes':
			st.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18)
			

		if child_everadpt != 'Select one':
			st.text("")
			st.write("Why did the child enter the foster care system? (Check all that apply)")
			
			col1, col2 = st.beta_columns(2)

			physical_abuse = col1.checkbox('Physical Abuse')
			sexual_abuse = col1.checkbox('Sexual Abuse')
			emotional_abuse_neglect = col1.checkbox('Emotional Abuse')
			physical_neglect = col1.checkbox("Physical Neglect")
			medical_neglect = col1.checkbox("Medical Neglect")
			alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse")
			drug_abuse_child = col1.checkbox("Child's Drug Abuse")
			child_behavior_problem = col1.checkbox('Child Behavior Problem')
			child_disability = col1.checkbox('Child Disability')
			transition_to_independence = col1.checkbox("Transition to Independence")
			inadequate_supervision = col1.checkbox("Inadequate Supervision")
			adoption_dissolution = col1.checkbox("Adoption Dissolution")
			abandonment = col1.checkbox("Abandonment")
			labor_trafficking = col1.checkbox("Labor Trafficking")
			sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation")
			
			prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse")
			prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse')
			prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse")
			prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect')
			prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect")
			alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse")
			drug_abuse_parent = col2.checkbox("Parent's Drug Abuse")
			incarceration_of_parent = col2.checkbox('Incarceration of Parent')
			death_of_parent = col2.checkbox('Death of Parent')
			domestic_violence = col2.checkbox("Domestic Violence")
			inadequate_housing = col2.checkbox("Inadequate Housing")
			caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope")
			relinquishment = col2.checkbox('Relinquishment')
			request_for_service = col2.checkbox('Request for Service')
			csec = col2.checkbox("CSEC")

			st.header("Current placement information")
			current_case_goal = st.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
			
		if current_case_goal != 'Select one':
			st.text("")
			st.write("Current placement's applicable payments")
			current_case_ivefc = st.checkbox("Foster Care Payments")
			current_case_iveaa = st.checkbox("Adoption Assistance")
			current_case_ivaafdc = st.checkbox("TANF Payment (Temporary Assistance for Needy Families")
			current_case_ivdchsup = st.checkbox("Child Support Funds")
			current_case_xixmedcd = st.checkbox("Medicaid")
			current_case_ssiother = st.checkbox("SSI or Social Security Benefits")
			current_case_noa = st.checkbox("Only State or Other Support")
			current_case_fcmntpay = st.number_input("Monthly Foster Care Payment ($)", min_value = 0, step = 100)


			st.text("")
			st.text("")
			find_providers_button = st.button("Find Providers")

		if find_providers_button:
			resetter = True

		if resetter == True:
			st.write('Child Birthday:', child_birthday)
			st.write('Child Race:', child_race)
			st.write('Child Hispanic?:', child_hispanic)
			st.write('Child Placed Before?:', placed_before)
			st.write('Number of Previous Placements:', num_prev_placements)
			st.write('Number of GOOD Previous Placements:', child_num_prev_placements_good)
			st.write('Number of BAD Previous Placements:', child_num_prev_placements_bad)
			st.write('Date of first placement:', child_date_of_first_placement)
			st.write('Most recent placement outcome:', child_recent_placement_outcome)
			st.write('Clincially diagnosed with disabilities?', child_clindis)
			st.write("Mental Retardation?:", child_mr_flag)
			st.write('Visually or Hearing Impaired?:', child_vishear_flag)
			st.write('Physically Disabled?:', child_phydis_flag)
			st.write('Emotionally Disturbed?:', child_emotdist_flag)
			st.write('Other Medically Diagnosed Condition?:', child_othermed_flag)
			st.write('Child ever adopted?:', child_everadpt)

			st.write('Physical Abuse:', physical_abuse)
			st.write('Sexual Abuse:', sexual_abuse)
			st.write('Emotional Abuse:', emotional_abuse_neglect)
			st.write("Physical Neglect:", physical_neglect)
			st.write("Medical Neglect:", medical_neglect)
			st.write("Child's Alcohol Abuse:", alcohol_abuse_child)
			st.write("Child's Drug Abuse:", drug_abuse_child)
			st.write('Child Behavior Problem:', child_behavior_problem)
			st.write('Child Disability:', child_disability)
			st.write("Transition to Independence:", transition_to_independence)
			st.write("Inadequate Supervision:", inadequate_supervision)
			st.write("Adoption Dissolution:", adoption_dissolution)
			st.write("Abandonment:", abandonment)
			st.write("Labor Trafficking:", labor_trafficking)
			st.write("Sexual Exploitation:", sexual_abuse_sexual_exploitation)
			st.write("Prospective Physical Abuse:", prospective_physical_abuse)
			st.write('Prospective Sexual Abuse:', prospective_sexual_abuse)
			st.write("Prospective Emotional Abuse:", prospective_emotional_abuse_neglect)
			st.write('Prospective Physical Neglect:', prospective_physical_neglect)
			st.write("Prospective Medical Neglect:", prospective_medical_neglect)
			st.write("Parent's Alcohol Abuse:", alcohol_abuse_parent)
			st.write("Parent's Drug Abuse:", drug_abuse_parent)
			st.write('Incarceration of Parent:', incarceration_of_parent)
			st.write('Death of Parent:', death_of_parent)
			st.write("Domestic Violence:", domestic_violence)
			st.write("Inadequate Housing:", inadequate_housing)
			st.write("Caregiver's inability to cope:", caregiver_inability_to_cope)
			st.write('Relinquishment:', relinquishment)
			st.write('Request for Service:', request_for_service)
			st.write("CSEC:", csec)

			st.write('Current Case Goal:', current_case_goal)
			st.write("Foster Care Payments:", current_case_ivefc)
			st.write("Adoption Assistance:", current_case_iveaa)
			st.write("TANF Payment (Temporary Assistance for Needy Families:", current_case_ivaafdc)
			st.write("Child Support Funds:", current_case_ivdchsup)
			st.write("Medicaid:", current_case_xixmedcd)
			st.write("SSI or Social Security Benefits:", current_case_ssiother)
			st.write("Only State or Other Support:", current_case_noa)
			st.write("Monthly Foster Care Payment ($):", current_case_fcmntpay)

			output = pd.read_csv('dummy_output.csv')
			st.title('Top Matched Providers')
			index = 0
			row = output.loc[0,:]
			st.write(str(index + 1),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			st.write("Match Rating: ", row.match_rating)
			st.write("Estimated Duration: ", row.estimated_duration)
			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			st.button("See other details")
			st.text('')

	# with col2.header:
	# 	st.subheader('child journey in faster care system')

	# with col2.graph:
	# 	st.markdown("""
	# 	there will be two graphs shown
	# 	""")

			# output = pd.read_csv('dummy_output.csv')
			# st.title('Top Matched Providers')
			# for index, row in output.iterrows():
			# 	st.write(str(index + 1),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			# 	st.write("Match Rating: ", row.match_rating)
			# 	st.write("Estimated Duration: ", row.estimated_duration)
			# 	st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			# 	st.button("See other details")
			# 	st.text('')

				# MATCH RATING MODEL
				## read in provider table (as needed by the match_rating model)
				#### will need other lookup tables as well
				## import match_rating model
				## match_rating model uses user input from above and provider table's provider features and predict rating on all providers
				## sort providers by rating, and take the top 10 (or top 100)


				# PLACEMENT DURATION MODEL
				## read in provider table (as needed by the placement_duration)
				## import placement_duration model
				## placement_duration model uses user input from above and top 10 provider features and predicts duration
				## attach placement duration prediction to record


				# POSITIVE OUTCOME PROBABILITY MODEL
				## read in provider table (as needed by the positive outcome probability)
				## import placement_duration model
				## placement_duration model uses user input from above and top 10 provider features and predicts duration
				## attach placement duration prediction to record


### DEMO PAGE ###
elif mypage == 'Matcher - Select Box':
	header = st.beta_container()
	product = st.beta_container()

	with header:
		# Creating the Titles and Image	
		st.title("Foster Care Matcher")
		st.header("Find the right foster care provider for your child")
		st.write("Use all of the existing available data on previous placements to find the best Provider whose suited to care for the new foster child")


	with product:
		## initialize values
		placed_before = 'Select one'
		num_prev_placements = 0
		child_num_prev_placements_good = 0
		child_num_prev_placements_bad = 0
		child_date_of_first_placement = datetime.date(9999,1,1)
		child_recent_placement_outcome = 'Select one'
		child_hispanic = 'Select one'
		child_mr_flag = False
		child_vishear_flag = False
		child_phydis_flag = False
		child_emotdist_flag = False
		child_othermed_flag = False
		child_clindis = 'Select one'
		child_everadpt = 'Select one'
		current_case_goal = 'Select one'
		find_providers_button = None
		find_providers_selection = 0

		placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

		if placed_before == 'Yes':
			num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

		if num_prev_placements > 0:
			st.header("Previous Placement Information")
			child_date_of_first_placement = st.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_num_prev_placements_good = st.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			child_num_prev_placements_bad = st.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			st.text("Remaining placements will be counted as having a NEUTRAL outcome.")
			st.text("")
			child_recent_placement_outcome = st.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])

		if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
			st.header("Child Information")
			child_birthday = st.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_race = st.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
			child_hispanic = st.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'])

		if child_hispanic != 'Select one':
			st.text("")
			st.write("Child's Disabilities")
			child_clindis = st.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])

		if child_clindis == 'Yes':
			st.write("Check all that apply:")
			child_mr_flag = st.checkbox("Mental Retardation")
			child_vishear_flag = st.checkbox("Visually or Hearing Impaired")
			child_phydis_flag = st.checkbox("Physically Disabled")
			child_emotdist_flag = st.checkbox("Emotionally Disturbed")
			child_othermed_flag = st.checkbox("Other Medically Diagnosed Condition")

		if child_clindis != 'Select one':
			st.text("")
			child_everadpt = st.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'])


		if child_everadpt == 'Yes':
			st.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18)
			

		if child_everadpt != 'Select one':
			st.text("")
			st.write("Why did the child enter the foster care system? (Check all that apply)")
			
			col1, col2 = st.beta_columns(2)

			physical_abuse = col1.checkbox('Physical Abuse')
			sexual_abuse = col1.checkbox('Sexual Abuse')
			emotional_abuse_neglect = col1.checkbox('Emotional Abuse')
			physical_neglect = col1.checkbox("Physical Neglect")
			medical_neglect = col1.checkbox("Medical Neglect")
			alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse")
			drug_abuse_child = col1.checkbox("Child's Drug Abuse")
			child_behavior_problem = col1.checkbox('Child Behavior Problem')
			child_disability = col1.checkbox('Child Disability')
			transition_to_independence = col1.checkbox("Transition to Independence")
			inadequate_supervision = col1.checkbox("Inadequate Supervision")
			adoption_dissolution = col1.checkbox("Adoption Dissolution")
			abandonment = col1.checkbox("Abandonment")
			labor_trafficking = col1.checkbox("Labor Trafficking")
			sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation")
			
			prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse")
			prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse')
			prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse")
			prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect')
			prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect")
			alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse")
			drug_abuse_parent = col2.checkbox("Parent's Drug Abuse")
			incarceration_of_parent = col2.checkbox('Incarceration of Parent')
			death_of_parent = col2.checkbox('Death of Parent')
			domestic_violence = col2.checkbox("Domestic Violence")
			inadequate_housing = col2.checkbox("Inadequate Housing")
			caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope")
			relinquishment = col2.checkbox('Relinquishment')
			request_for_service = col2.checkbox('Request for Service')
			csec = col2.checkbox("CSEC")

			st.header("Current placement information")
			current_case_goal = st.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
			
		if current_case_goal != 'Select one':
			st.text("")
			st.write("Current placement's applicable payments")
			current_case_ivefc = st.checkbox("Foster Care Payments")
			current_case_iveaa = st.checkbox("Adoption Assistance")
			current_case_ivaafdc = st.checkbox("TANF Payment (Temporary Assistance for Needy Families")
			current_case_ivdchsup = st.checkbox("Child Support Funds")
			current_case_xixmedcd = st.checkbox("Medicaid")
			current_case_ssiother = st.checkbox("SSI or Social Security Benefits")
			current_case_noa = st.checkbox("Only State or Other Support")
			current_case_fcmntpay = st.number_input("Monthly Foster Care Payment ($)", min_value = 0, step = 100)


			st.text("")
			st.text("")
			find_providers_selection = st.selectbox("Show Top Matched Providers:", [0,5,10,15,20])

		if find_providers_selection > 0:
			st.write('Child Birthday:', child_birthday)
			st.write('Child Race:', child_race)
			st.write('Child Hispanic?:', child_hispanic)
			st.write('Child Placed Before?:', placed_before)
			st.write('Number of Previous Placements:', num_prev_placements)
			st.write('Number of GOOD Previous Placements:', child_num_prev_placements_good)
			st.write('Number of BAD Previous Placements:', child_num_prev_placements_bad)
			st.write('Date of first placement:', child_date_of_first_placement)
			st.write('Most recent placement outcome:', child_recent_placement_outcome)
			st.write('Clincially diagnosed with disabilities?', child_clindis)
			st.write("Mental Retardation?:", child_mr_flag)
			st.write('Visually or Hearing Impaired?:', child_vishear_flag)
			st.write('Physically Disabled?:', child_phydis_flag)
			st.write('Emotionally Disturbed?:', child_emotdist_flag)
			st.write('Other Medically Diagnosed Condition?:', child_othermed_flag)
			st.write('Child ever adopted?:', child_everadpt)

			st.write('Physical Abuse:', physical_abuse)
			st.write('Sexual Abuse:', sexual_abuse)
			st.write('Emotional Abuse:', emotional_abuse_neglect)
			st.write("Physical Neglect:", physical_neglect)
			st.write("Medical Neglect:", medical_neglect)
			st.write("Child's Alcohol Abuse:", alcohol_abuse_child)
			st.write("Child's Drug Abuse:", drug_abuse_child)
			st.write('Child Behavior Problem:', child_behavior_problem)
			st.write('Child Disability:', child_disability)
			st.write("Transition to Independence:", transition_to_independence)
			st.write("Inadequate Supervision:", inadequate_supervision)
			st.write("Adoption Dissolution:", adoption_dissolution)
			st.write("Abandonment:", abandonment)
			st.write("Labor Trafficking:", labor_trafficking)
			st.write("Sexual Exploitation:", sexual_abuse_sexual_exploitation)
			st.write("Prospective Physical Abuse:", prospective_physical_abuse)
			st.write('Prospective Sexual Abuse:', prospective_sexual_abuse)
			st.write("Prospective Emotional Abuse:", prospective_emotional_abuse_neglect)
			st.write('Prospective Physical Neglect:', prospective_physical_neglect)
			st.write("Prospective Medical Neglect:", prospective_medical_neglect)
			st.write("Parent's Alcohol Abuse:", alcohol_abuse_parent)
			st.write("Parent's Drug Abuse:", drug_abuse_parent)
			st.write('Incarceration of Parent:', incarceration_of_parent)
			st.write('Death of Parent:', death_of_parent)
			st.write("Domestic Violence:", domestic_violence)
			st.write("Inadequate Housing:", inadequate_housing)
			st.write("Caregiver's inability to cope:", caregiver_inability_to_cope)
			st.write('Relinquishment:', relinquishment)
			st.write('Request for Service:', request_for_service)
			st.write("CSEC:", csec)

			st.write('Current Case Goal:', current_case_goal)
			st.write("Foster Care Payments:", current_case_ivefc)
			st.write("Adoption Assistance:", current_case_iveaa)
			st.write("TANF Payment (Temporary Assistance for Needy Families:", current_case_ivaafdc)
			st.write("Child Support Funds:", current_case_ivdchsup)
			st.write("Medicaid:", current_case_xixmedcd)
			st.write("SSI or Social Security Benefits:", current_case_ssiother)
			st.write("Only State or Other Support:", current_case_noa)
			st.write("Monthly Foster Care Payment ($):", current_case_fcmntpay)

			output = pd.read_csv('dummy_output.csv')
			st.title('Top Matched Providers')
			index = 0
			row = output.loc[0,:]
			st.write(str(index + 1),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			st.write("Match Rating: ", row.match_rating)
			st.write("Estimated Duration: ", row.estimated_duration)
			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			new_details_button = st.button("See other details")

			if new_details_button:
				st.text('Showing distribution of placement end reasons')

			st.write(str(index + 2),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			st.write("Match Rating: ", row.match_rating)
			st.write("Estimated Duration: ", row.estimated_duration)
			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			new_details_button = st.button("See other details 2")

			if new_details_button:
				st.text('Showing distribution of placement end reasons')


### DEMO PAGE ###
elif mypage == 'Matcher - Form Button':
	header = st.beta_container()
	product = st.beta_container()

	with header:
		# Creating the Titles and Image	
		st.title("Foster Care Matcher")
		st.header("Find the right foster care provider for your child")
		st.write("Use all of the existing available data on previous placements to find the best Provider whose suited to care for the new foster child")


	with product:
		## initialize values
		placed_before = 'Select one'
		num_prev_placements = 0
		child_num_prev_placements_good = 0
		child_num_prev_placements_bad = 0
		child_date_of_first_placement = datetime.date(9999,1,1)
		child_recent_placement_outcome = 'Select one'
		child_hispanic = 'Select one'
		child_mr_flag = False
		child_vishear_flag = False
		child_phydis_flag = False
		child_emotdist_flag = False
		child_othermed_flag = False
		child_clindis = 'Select one'
		child_everadpt = 'Select one'
		current_case_goal = 'Select one'
		find_providers_button = None
		find_providers_selection = 0
		find_providers_form_button = None

		placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

		if placed_before == 'Yes':
			num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

		if num_prev_placements > 0:
			st.header("Previous Placement Information")
			child_date_of_first_placement = st.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_num_prev_placements_good = st.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			child_num_prev_placements_bad = st.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
			st.text("Remaining placements will be counted as having a NEUTRAL outcome.")
			st.text("")
			child_recent_placement_outcome = st.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])

		if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
			st.header("Child Information")
			child_birthday = st.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
			child_race = st.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
			child_hispanic = st.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'])

		if child_hispanic != 'Select one':
			st.text("")
			st.write("Child's Disabilities")
			child_clindis = st.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])

		if child_clindis == 'Yes':
			st.write("Check all that apply:")
			child_mr_flag = st.checkbox("Mental Retardation")
			child_vishear_flag = st.checkbox("Visually or Hearing Impaired")
			child_phydis_flag = st.checkbox("Physically Disabled")
			child_emotdist_flag = st.checkbox("Emotionally Disturbed")
			child_othermed_flag = st.checkbox("Other Medically Diagnosed Condition")

		if child_clindis != 'Select one':
			st.text("")
			child_everadpt = st.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'])


		if child_everadpt == 'Yes':
			st.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18)
			

		if child_everadpt != 'Select one':
			st.text("")
			st.write("Why did the child enter the foster care system? (Check all that apply)")
			
			col1, col2 = st.beta_columns(2)

			physical_abuse = col1.checkbox('Physical Abuse')
			sexual_abuse = col1.checkbox('Sexual Abuse')
			emotional_abuse_neglect = col1.checkbox('Emotional Abuse')
			physical_neglect = col1.checkbox("Physical Neglect")
			medical_neglect = col1.checkbox("Medical Neglect")
			alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse")
			drug_abuse_child = col1.checkbox("Child's Drug Abuse")
			child_behavior_problem = col1.checkbox('Child Behavior Problem')
			child_disability = col1.checkbox('Child Disability')
			transition_to_independence = col1.checkbox("Transition to Independence")
			inadequate_supervision = col1.checkbox("Inadequate Supervision")
			adoption_dissolution = col1.checkbox("Adoption Dissolution")
			abandonment = col1.checkbox("Abandonment")
			labor_trafficking = col1.checkbox("Labor Trafficking")
			sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation")
			
			prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse")
			prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse')
			prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse")
			prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect')
			prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect")
			alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse")
			drug_abuse_parent = col2.checkbox("Parent's Drug Abuse")
			incarceration_of_parent = col2.checkbox('Incarceration of Parent')
			death_of_parent = col2.checkbox('Death of Parent')
			domestic_violence = col2.checkbox("Domestic Violence")
			inadequate_housing = col2.checkbox("Inadequate Housing")
			caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope")
			relinquishment = col2.checkbox('Relinquishment')
			request_for_service = col2.checkbox('Request for Service')
			csec = col2.checkbox("CSEC")

			st.header("Current placement information")
			current_case_goal = st.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
			
		if current_case_goal != 'Select one':
			st.text("")
			st.write("Current placement's applicable payments")
			current_case_ivefc = st.checkbox("Foster Care Payments")
			current_case_iveaa = st.checkbox("Adoption Assistance")
			current_case_ivaafdc = st.checkbox("TANF Payment (Temporary Assistance for Needy Families")
			current_case_ivdchsup = st.checkbox("Child Support Funds")
			current_case_xixmedcd = st.checkbox("Medicaid")
			current_case_ssiother = st.checkbox("SSI or Social Security Benefits")
			current_case_noa = st.checkbox("Only State or Other Support")
			current_case_fcmntpay = st.number_input("Monthly Foster Care Payment ($)", min_value = 0, step = 100)


			st.text("")
			st.text("")
			find_providers_form = st.form(key = 'find_providers_form')
			find_providers_form_button = find_providers_form.form_submit_button(label = 'Find Providers')


		if find_providers_form_button:
			st.write('Child Birthday:', child_birthday)
			st.write('Child Race:', child_race)
			st.write('Child Hispanic?:', child_hispanic)
			st.write('Child Placed Before?:', placed_before)
			st.write('Number of Previous Placements:', num_prev_placements)
			st.write('Number of GOOD Previous Placements:', child_num_prev_placements_good)
			st.write('Number of BAD Previous Placements:', child_num_prev_placements_bad)
			st.write('Date of first placement:', child_date_of_first_placement)
			st.write('Most recent placement outcome:', child_recent_placement_outcome)
			st.write('Clincially diagnosed with disabilities?', child_clindis)
			st.write("Mental Retardation?:", child_mr_flag)
			st.write('Visually or Hearing Impaired?:', child_vishear_flag)
			st.write('Physically Disabled?:', child_phydis_flag)
			st.write('Emotionally Disturbed?:', child_emotdist_flag)
			st.write('Other Medically Diagnosed Condition?:', child_othermed_flag)
			st.write('Child ever adopted?:', child_everadpt)

			st.write('Physical Abuse:', physical_abuse)
			st.write('Sexual Abuse:', sexual_abuse)
			st.write('Emotional Abuse:', emotional_abuse_neglect)
			st.write("Physical Neglect:", physical_neglect)
			st.write("Medical Neglect:", medical_neglect)
			st.write("Child's Alcohol Abuse:", alcohol_abuse_child)
			st.write("Child's Drug Abuse:", drug_abuse_child)
			st.write('Child Behavior Problem:', child_behavior_problem)
			st.write('Child Disability:', child_disability)
			st.write("Transition to Independence:", transition_to_independence)
			st.write("Inadequate Supervision:", inadequate_supervision)
			st.write("Adoption Dissolution:", adoption_dissolution)
			st.write("Abandonment:", abandonment)
			st.write("Labor Trafficking:", labor_trafficking)
			st.write("Sexual Exploitation:", sexual_abuse_sexual_exploitation)
			st.write("Prospective Physical Abuse:", prospective_physical_abuse)
			st.write('Prospective Sexual Abuse:', prospective_sexual_abuse)
			st.write("Prospective Emotional Abuse:", prospective_emotional_abuse_neglect)
			st.write('Prospective Physical Neglect:', prospective_physical_neglect)
			st.write("Prospective Medical Neglect:", prospective_medical_neglect)
			st.write("Parent's Alcohol Abuse:", alcohol_abuse_parent)
			st.write("Parent's Drug Abuse:", drug_abuse_parent)
			st.write('Incarceration of Parent:', incarceration_of_parent)
			st.write('Death of Parent:', death_of_parent)
			st.write("Domestic Violence:", domestic_violence)
			st.write("Inadequate Housing:", inadequate_housing)
			st.write("Caregiver's inability to cope:", caregiver_inability_to_cope)
			st.write('Relinquishment:', relinquishment)
			st.write('Request for Service:', request_for_service)
			st.write("CSEC:", csec)

			st.write('Current Case Goal:', current_case_goal)
			st.write("Foster Care Payments:", current_case_ivefc)
			st.write("Adoption Assistance:", current_case_iveaa)
			st.write("TANF Payment (Temporary Assistance for Needy Families:", current_case_ivaafdc)
			st.write("Child Support Funds:", current_case_ivdchsup)
			st.write("Medicaid:", current_case_xixmedcd)
			st.write("SSI or Social Security Benefits:", current_case_ssiother)
			st.write("Only State or Other Support:", current_case_noa)
			st.write("Monthly Foster Care Payment ($):", current_case_fcmntpay)

			output = pd.read_csv('dummy_output.csv')
			st.title('Top Matched Providers')
			index = 0
			row = output.loc[0,:]
			st.write(str(index + 1),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			st.write("Match Rating: ", row.match_rating)
			st.write("Estimated Duration: ", row.estimated_duration)
			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			new_details_button = st.button("See other details")

			if new_details_button:
				st.text('Showing distribution of placement end reasons')

			st.write(str(index + 2),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
			st.write("Match Rating: ", row.match_rating)
			st.write("Estimated Duration: ", row.estimated_duration)
			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
			new_details_button = st.button("See other details 2")

			if new_details_button:
				st.text('Showing distribution of placement end reasons')


### ARCHITECTURE PAGE ###
elif mypage == 'Architecture':
	# set the page layout
	# st.set_page_config(layout="wide")

	header = st.beta_container()
	product = st.beta_container()

	with header:
		st.title('Foster Care Matcher')
		st.header('Features about Foster Care Matcher')
		st.subheader('this is data pipeline and ML pipeline')
		image = Image.open('pipeline_mk2.png').convert('RGB').save('pipeline_mk_new.png')
		image = Image.open('pipeline_mk_new.png')
		st.image(image, width = 800)
		# st.title('Foster Care Matcher')
		# st.header('Features about Foster Care Matcher')
		# st.subheader('this is data pipeline and ML pipeline')
		# st.write('Process on creating this')


	with product:
		st.subheader('this is the past journey of this child through foster care system ')
		df = pd. DataFrame(
		np.random.randn(200, 3),
		columns=['a', 'b', 'c'])

		c = alt.Chart(df).mark_circle().encode(
			x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
	
		st.write(c)

### MODELING PAGE ###
elif mypage == 'Modeling':
	st.title('Foster Care Matcher')
	st.header('Features about Foster Care Matcher')
	st.write('Process on creating this')

	
### TEAM PAGE ###
elif mypage == 'Team':
	st.title('Foster Care Matcher')
	st.header('Who worked on Information')
	st.write('Pictures')
	st.title("We'd like to thank")
	st.header('Orgs')
	st.write('Robert and Roshannon, David and Joyce')	




### TEST PAGE ###
# elif mypage == 'TEST':
# 	st.title('Foster Care Matcher')
# 	st.header('TESTING PAGE')
# 	num_prev_placements = 0
# 	placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

# 	if placed_before == 'Yes':
# 		num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

# 	if num_prev_placements > 0:
# 		yes_prev_placement_form = st.form(key='yes_prev_placement_form')

# 		yes_prev_placement_form.header("Previous placement information")
# 		child_date_of_first_placement = yes_prev_placement_form.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
# 		child_num_prev_placements_good = yes_prev_placement_form.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
# 		child_num_prev_placements_bad = yes_prev_placement_form.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
# 		yes_prev_placement_form.text("Remaining placements will be counted as having a NEUTRAL outcome.")
# 		child_recent_placement_outcome = yes_prev_placement_form.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])


# 		yes_prev_placement_form.header("Child information")
# 		child_birthday = yes_prev_placement_form.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
# 		child_race = yes_prev_placement_form.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
# 		child_hispanic_flag = yes_prev_placement_form.checkbox("Is the child Hispanic?")
# 		yes_prev_placement_form.text("")
# 		yes_prev_placement_form.write("Child's disabilities (Check all that apply)")
# 		# child_clindis = yes_prev_placement_form.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])
# 		child_mr_flag = yes_prev_placement_form.checkbox("Mental Retardation")
# 		child_vishear_flag = yes_prev_placement_form.checkbox("Visually or Hearing Impaired")
# 		child_phydis_flag = yes_prev_placement_form.checkbox("Physically Disabled")
# 		child_emotdist_flag = yes_prev_placement_form.checkbox("Emotionally Disturbed")
# 		child_othermed_flag = yes_prev_placement_form.checkbox("Other Medically Diagnosed Condition")
# 		yes_prev_placement_form.text("")
# 		yes_prev_placement_form.selectbox("Has the child ever been adopted?", ['Select box', 'Yes', 'No'])
# 		yes_prev_placement_form.slider("How old was the child at the time of their most recent adoption?", min_value=0, max_value=18)
# 		yes_prev_placement_form.text("")
# 		yes_prev_placement_form.write("What were the reasons for the child to initially enter the foster care system? (Check all that apply)")
		
# 		col1, col2, col3 = yes_prev_placement_form.beta_columns(3)

# 		physical_abuse = col1.checkbox('Physical Abuse')
# 		sexual_abuse = col1.checkbox('Sexual Abuse')
# 		emotional_abuse_neglect = col1.checkbox('Emotional Abuse or Neglect')
# 		alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse")
# 		drug_abuse_child = col1.checkbox("Child's Drug Abuse")
# 		alcohol_abuse_parent = col1.checkbox("Parent's Alcohol Abuse")
# 		drug_abuse_parent = col1.checkbox("Parent's Drug Abuse")
# 		physical_neglect = col1.checkbox("Physical Neglect")
# 		domestic_violence = col1.checkbox("Domestic Violence")
# 		inadequate_housing = col1.checkbox("Inadequate Housing")

# 		child_behavior_problem = col2.checkbox('Child Behavior Problem')
# 		child_disability = col2.checkbox('Child Disability')
# 		incarceration_of_parent = col2.checkbox('Incarceration of Parent')
# 		death_of_parent = col2.checkbox('Death of Parent')
# 		caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope")
# 		abandonment = col2.checkbox("Abandonment")
# 		transition_to_independence = col2.checkbox("Transition to Independence")
# 		inadequate_supervision = col2.checkbox("Inadequate Supervision")
# 		prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse")
# 		prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect")

# 		prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse")
# 		prospective_physical_neglect = col3.checkbox('Prospective Physical Neglect')
# 		prospective_sexual_abuse = col3.checkbox('Prospective Sexual Abuse')
# 		relinquishment = col3.checkbox('Relinquishment')
# 		request_for_service = col3.checkbox('Request for Service')
# 		adoption_dissolution = col3.checkbox("Adoption Dissolution")
# 		medical_neglect = col3.checkbox("Medical Neglect")
# 		csec = col3.checkbox("CSEC")
# 		labor_trafficking = col3.checkbox("Labor Trafficking")
# 		sexual_abuse_sexual_exploitation = col3.checkbox("Sexual Abuse/Exploitation")



# 		yes_prev_placement_form.header("Current placement information")
# 		current_case_goal = yes_prev_placement_form.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
# 		yes_prev_placement_form.text("")
# 		yes_prev_placement_form.write("Current placement's applicable payments")
# 		current_case_ivefc = yes_prev_placement_form.checkbox("Foster Care Payments")
# 		current_case_iveaa = yes_prev_placement_form.checkbox("Adoption Assistance")
# 		current_case_ivaafdc = yes_prev_placement_form.checkbox("TANF Payment (Temporary Assistance for Needy Families")
# 		current_case_ivdchsup = yes_prev_placement_form.checkbox("Child Support Funds")
# 		current_case_xixmedcd = yes_prev_placement_form.checkbox("Medicaid")
# 		current_case_ssiother = yes_prev_placement_form.checkbox("SSI or Social Security Benefits")
# 		current_case_noa = yes_prev_placement_form.checkbox("Only State or Other Support")
# 		current_case_fcmntpay = yes_prev_placement_form.number_input("Monthly Foster Care Payment ($)", min_value = 0.00)

# 		yes_prev_placement_submit_button = yes_prev_placement_form.form_submit_button(label='Submit')

	
# elif mypage == 'TEST_NOFORM':
# 	st.title('Foster Care Matcher')
# 	placed_before = 'Select one'
# 	num_prev_placements = 0
# 	child_recent_placement_outcome = 'Select one'
# 	child_hispanic = 'Select one'
# 	child_clindis = 'Select one'
# 	child_everadpt = 'Select one'
# 	current_case_goal = 'Select one'
# 	find_providers_button = None

# 	placed_before = st.selectbox("Has this child been placed before?", ['Select one', 'Yes', 'No'])

# 	if placed_before == 'Yes':
# 		num_prev_placements = st.number_input('How many previous placements has this child had?', min_value = 0, max_value = 100, step = 1)

# 	if num_prev_placements > 0:
# 		st.header("Previous Placement Information")
# 		child_date_of_first_placement = st.date_input("What was the start date for the very first placement?", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
# 		child_num_prev_placements_good = st.number_input('Out of the total previous placements, how many of them had a POSITIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
# 		child_num_prev_placements_bad = st.number_input('Out of the total previous placements, how many of them had a NEGATIVE outcome?', min_value = 0, max_value = num_prev_placements, step = 1)
# 		st.text("Remaining placements will be counted as having a NEUTRAL outcome.")
# 		st.text("")
# 		child_recent_placement_outcome = st.selectbox("What was the outcome of the child's most recent placement?", ['Select one', 'Positive', 'Neutral', 'Negative'])

# 	if child_recent_placement_outcome != 'Select one' or placed_before == 'No':
# 		st.header("Child Information")
# 		child_birthday = st.date_input("Child's birthday", datetime.date(2015,1,1), min_value = (datetime.datetime.now() - datetime.timedelta(days = 6570)), max_value = datetime.datetime.now())
# 		child_race = st.selectbox("Child's Race", ['Select one', 'White', 'Black', 'Asian', 'Pacific Islander', 'Native American', 'Multi-Racial'])
# 		child_hispanic = st.selectbox("Is the child Hispanic?", ['Select one', 'Yes', 'No'])

# 	if child_hispanic != 'Select one':
# 		st.text("")
# 		st.write("Child's Disabilities")
# 		child_clindis = st.selectbox("Has the child been clinically diagnosed with disabilities?", ['Select one', 'Yes', 'No', 'Not yet determined'])

# 	if child_clindis == 'Yes':
# 		st.write("Check all that apply:")
# 		child_mr_flag = st.checkbox("Mental Retardation")
# 		child_vishear_flag = st.checkbox("Visually or Hearing Impaired")
# 		child_phydis_flag = st.checkbox("Physically Disabled")
# 		child_emotdist_flag = st.checkbox("Emotionally Disturbed")
# 		child_othermed_flag = st.checkbox("Other Medically Diagnosed Condition")

# 	if child_clindis != 'Select one':
# 		st.text("")
# 		child_everadpt = st.selectbox("Has the child ever been adopted?", ['Select one', 'Yes', 'No'])


# 	if child_everadpt == 'Yes':
# 		st.slider("How old was the child at the time of their most recent adoption? (Years)", min_value=0, max_value=18)
		

# 	if child_everadpt != 'Select one':
# 		st.text("")
# 		st.write("Why did the child enter the foster care system? (Check all that apply)")
		
# 		col1, col2 = st.beta_columns(2)

# 		physical_abuse = col1.checkbox('Physical Abuse')
# 		sexual_abuse = col1.checkbox('Sexual Abuse')
# 		emotional_abuse_neglect = col1.checkbox('Emotional Abuse')
# 		physical_neglect = col1.checkbox("Physical Neglect")
# 		medical_neglect = col1.checkbox("Medical Neglect")
# 		alcohol_abuse_child = col1.checkbox("Child's Alcohol Abuse")
# 		drug_abuse_child = col1.checkbox("Child's Drug Abuse")
# 		child_behavior_problem = col1.checkbox('Child Behavior Problem')
# 		child_disability = col1.checkbox('Child Disability')
# 		transition_to_independence = col1.checkbox("Transition to Independence")
# 		inadequate_supervision = col1.checkbox("Inadequate Supervision")
# 		adoption_dissolution = col1.checkbox("Adoption Dissolution")
# 		abandonment = col1.checkbox("Abandonment")
# 		labor_trafficking = col1.checkbox("Labor Trafficking")
# 		sexual_abuse_sexual_exploitation = col1.checkbox("Sexual Exploitation")
		
# 		prospective_physical_abuse = col2.checkbox("Prospective Physical Abuse")
# 		prospective_sexual_abuse = col2.checkbox('Prospective Sexual Abuse')
# 		prospective_emotional_abuse_neglect = col2.checkbox("Prospective Emotional Abuse")
# 		prospective_physical_neglect = col2.checkbox('Prospective Physical Neglect')
# 		prospective_medical_neglect = col2.checkbox("Prospective Medical Neglect")
# 		alcohol_abuse_parent = col2.checkbox("Parent's Alcohol Abuse")
# 		drug_abuse_parent = col2.checkbox("Parent's Drug Abuse")
# 		incarceration_of_parent = col2.checkbox('Incarceration of Parent')
# 		death_of_parent = col2.checkbox('Death of Parent')
# 		domestic_violence = col2.checkbox("Domestic Violence")
# 		inadequate_housing = col2.checkbox("Inadequate Housing")
# 		caregiver_inability_to_cope = col2.checkbox("Caregiver's inability to cope")
# 		relinquishment = col2.checkbox('Relinquishment')
# 		request_for_service = col2.checkbox('Request for Service')
# 		csec = col2.checkbox("CSEC")

# 		st.header("Current placement information")
# 		current_case_goal = st.selectbox("What is the goal for this placement based on the child's case plan?", ['Select one', 'Reunification', 'Live with Other Relatives', 'Adoption', 'Long Term Foster Care', 'Emancipation', 'Guardianship', 'Goal Not Yet Established'])
		
# 	if current_case_goal != 'Select one':
# 		st.text("")
# 		st.write("Current placement's applicable payments")
# 		current_case_ivefc = st.checkbox("Foster Care Payments")
# 		current_case_iveaa = st.checkbox("Adoption Assistance")
# 		current_case_ivaafdc = st.checkbox("TANF Payment (Temporary Assistance for Needy Families")
# 		current_case_ivdchsup = st.checkbox("Child Support Funds")
# 		current_case_xixmedcd = st.checkbox("Medicaid")
# 		current_case_ssiother = st.checkbox("SSI or Social Security Benefits")
# 		current_case_noa = st.checkbox("Only State or Other Support")
# 		current_case_fcmntpay = st.number_input("Monthly Foster Care Payment ($)", min_value = 0, step = 100)


# 		st.text("")
# 		st.text("")
# 		find_providers_button = st.button("Find Providers")

# 	if find_providers_button:





# 		output = pd.read_csv('dummy_output.csv')
# 		st.title('Top Matched Providers')
# 		for index, row in output.iterrows():
# 			st.write(str(index + 1),". ", row.provider_name, '    (Provider ID: ', row.provider_id, ")")
# 			st.write("Match Rating: ", row.match_rating)
# 			st.write("Estimated Duration: ", row.estimated_duration)
# 			st.write("Most Likely End Reason: ", row.most_likely_end_reason)
# 			st.button("See other details")
# 			st.text('')
	
		


	# if placed_before_yes:
	# 	st.header("Child's Information")
	# 	st.write("Provide some details on the child you're looking to place:")

	# 	myform = st.form(key='myform')

	# 	child_age = myform.slider("Child's Age (years)", min_value=0, max_value=17)
	# 	child_race = myform.selectbox("Child's Race", ['White', 'Black', 'Asian', 'Pacific Islander'])
		
	# 	col1, col2 = myform.beta_columns(2)

	# 	col1.write("Child's Physical Disabilities")
	# 	child_visual_imp = col1.checkbox('Visual Impairment')
	# 	child_hearing_imp = col1.checkbox('Hearing Impairment')
	# 	child_speech_imp = col1.checkbox('Speech Impairment')
	# 	child_movement_imp = col1.checkbox('Movement Impairment')
	# 	child_other_physical = col1.checkbox('Other Physical Disability')

	# 	col2.write("Child's Mental Disabilities")
	# 	child_depression = col2.checkbox('Depression')
	# 	child_bipolar = col2.checkbox('Bipolar Disorder')
	# 	child_psychoses = col2.checkbox('Psychoses (Schizophrenia, etc.)')
	# 	child_autism = col2.checkbox('Autism')
	# 	child_other_mental = col2.checkbox('Other Mental Disability')
		
	# 	submit_button = myform.form_submit_button(label='Submit')


	# 	if submit_button:
	# 		## show top ten rows of matches
	# 		st.write("Child's Age: ", child_age)
	# 		st.write("Child's Race: ", child_race)
	# 		st.write("Child's Physical Disabilities: ", sum([child_visual_imp ,child_hearing_imp, child_speech_imp, child_movement_imp, child_other_physical]))
	# 		st.write("Child's Mental Disabilities: ", sum([child_depression, child_bipolar, child_psychoses, child_autism, child_other_mental]))
	# 		st.title('')

			
	# 		output = pd.read_csv('dummy_output.csv')

	# if placed_before_yes:
	# 	st.session_state.placed_before_yes = 1
	# 	placed_before_yes_form = st.form(key='placed_before_yes_form')

	# 	child_age = placed_before_yes_form.slider("Child's Age (years)", min_value=0, max_value=17)
	# 	child_race = placed_before_yes_form.selectbox("Child's Race", ['White', 'Black', 'Asian', 'Pacific Islander'])
		
	# 	placed_before_yes_form_submit_button = placed_before_yes_form.form_submit_button(label='Submit')

	# 	if placed_before_yes_form_submit_button:
	# 		st.write("YOU HIT THE YES BUTTON")

	# if placed_before_no:
	# 	placed_before_no_form = st.form(key='placed_before_no_form')

	# 	child_age = placed_before_no_form.slider("Child's Age (years)", min_value=0, max_value=17)
	# 	child_race = placed_before_no_form.selectbox("Child's Race", ['White', 'Black', 'Asian', 'Pacific Islander'])
		
	# 	placed_before_no_form_submit_button = placed_before_no_form.form_submit_button(label='placed_before_no_form_submit')



	# if placed_before_no_form_submit_button:
	# 	st.write("YOU HIT THE NO BUTTON")


# with footer:
# 	st.button("Start Over", on_click = reset_session())


# st.title('Counter Example')
# if 'count' not in st.session_state:
# 	st.session_state.count = 0

# increment = st.button('Increment')
# if increment:
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)


# # Instantiating a new DF row to classify later
# new_profile = pd.DataFrame()


# def reset_session():
# 	for key in st.session_state.keys():
# 		del st.session_state[key]

# Check if 'key' already exists in session_state
# If not, then initialize it
# if 'child_age' not in st.session_state:
# 	st.session_state['child_age'] = 0

# if 'child_race' not in st.session_state:
# 	st.session_state['child_race'] = 'White'

# if 'child_physical_disabilities' not in st.session_state:
# 	st.session_state['child_physical_disabilities'] = []

# if 'child_mental_disabilities' not in st.session_state:
# 	st.session_state['child_mental_disabilities'] = []


# if submit_button:
		# 	st.write("Child's Age: ", st.session_state['child_age'])
		# 	st.write("Child's Race: ", st.session_state['child_race'])







