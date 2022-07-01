"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os
from PIL import Image
from streamlit_option_menu import option_menu

# Data dependencies
import pandas as pd

#Vectorizer
news_vectorizer = open("resources/vector2.pkl","rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(news_vectorizer)

# Load your raw data
raw = pd.read_csv("resources/train.csv")
image = Image.open('IMG-20220629-WA0005.jpg')
st.set_page_config(page_title="Tweet Classifier App", page_icon=image)

#st.title("Streamlit Dashboard Demo")

# The main function where we will build the actual app
def main() -> object:
	global model

	# Creates a main title and subheader on your page -
	# these are static across all pages

	with st.sidebar:
		selected = option_menu(
			menu_title="Main Menu",
			options = ["Prediction", "Load a dataset", "About Us", "Contact Us", "Documentation"],
			icons=["emoji-expressionless","calendar", "people", "phone", "book"],
			menu_icon = "cast",
			default_index = 0
		)

	#st.title("Tweet Classifer")
	#st.subheader("Climate change tweet classification")


	# Building out the "Information" page
	if selected == "Load a dataset":
		image = Image.open('IMG-20220629-WA0005.jpg')
		st.image(image, width=150)

		st.info("You can upload a csv containing the many tweet messages for batch processing. To upload multiple files, please ensure that the files are in the following format message, tweetid ")
		# You can read a markdown file from supporting resources folder
		#st.markdown("Some information here")

		uploaded_file = st.file_uploader(label="Upload your csv or excel file", type=['csv', 'xlsx'])
		#Load a file to the web application
		global df
		if uploaded_file is not None:
			try:
				df = pd.read_csv(uploaded_file)
			except:
				df = pd.read_excel(uploaded_file)



		#st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
		#st.write(raw[['sentiment', 'message']]) # will write the df to the page
			try:
				st.write(df)
			except:
				st.write("Upload a file!")

	# Building out the predication page
	if selected == "Prediction":
		image = Image.open('IMG-20220629-WA0005.jpg')
		st.image(image, width=150)
		#st.title("Text Sentiment Analysis Prediction")
		#st.info("Prediction with ML Models")

		option = st.selectbox(
			'SELECT YOUR PREFERRED MODEL FROM THE DROPDOWN',
			('Linear Support Vector Classifier', 'Logistic Regression', 'Support Vector Classifier'))

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text", "Type Here")

		# Model selection options
		if option == 'Linear Support Vector Classifier':
			model = 'resources/naive_bayes_model.pkl'

		elif option == "Support Vector Classifier":
			model = 'resources/SVC2_model.pkl'

		elif option == 'Logistic Regression':
			model = 'resources/log_reg_model.pkl'

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)

			word = ''
			if prediction == 0:
				word = ' "**PRO:**".  This tweet neither supports nor refutes the belief of man-made climate change'

			elif prediction == 1:
				word = '"**PRO:**". This tweet supports the belief of man-made climate change'

			elif prediction == 2:
				word = '"**NEWS:**". This tweet links to factual news about climate change'

			else:
				word = '"**ANTI:**" This tweet does not believe in man-made climate change'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(word))

	if selected == "About Us":
		image = Image.open('Meet the team.jpg')
		st.image(image)
		text = """
		

 
Data fluent inc. is a world-class data science company that specializes 
in extracting useful insights from all kinds of raw data. 
Data collection and analysis are increasingly becoming very useful in industries and economies worldwide. 
With advances in science and technology (particularly information technology), we are in an age 
where an astounding quantity of data in many different forms is generated every second. 
This data usually have hidden within them insights on trends, habits, developments, changes, etc that may not be immediately identified.


Before now, it was near impossible to process these large swaths of data in order to reveal these insights. With the developments in the field 
of data science and the expertise of a company like ours,
 these data can be processed to not only reveal the insight hidden in them but to also present the discoveries 
 made in the process in a form that is digestible by a non-technical audience. Our team is made up of 5 highly qualified professionals 
 who excel in the fields of **Business Management, marketing and promotions, technical data science, IT communications, 
 and Administration**. Please refer to this link to access our full company profile. 
		
		"""
		st.markdown("<h2 style='text-align: center; color: blue;'>About Data fluent Inc.</h2>", unsafe_allow_html=True)
		st.markdown(text )


	if selected == "Documentation":
		image = Image.open('New Cover page.png')
		st.image(image)

		st.markdown("<h2 style='text-align: center; color: blue;'>Documentation Page Info</h2>", unsafe_allow_html=True)

		col1, col2, col3 = st.columns(3)

		with col1:
			#st.markdown("<h2 style='text-align: center; color: black;'>Overview</h2>", unsafe_allow_html=True)
			text1 = """
<h3>Overview</h3> 			
<h4 style="text-align:justify; line-height:150%; color: black; ">
			
 
The emphasis on the burden of climate change has been constantly expanding to include personal responsibility rather 
than general by huge companies and Governments alone. It is increasingly obvious that action needs to be taken on 
personal levels to combat climate change and its attendant effects. Many companies are built around lessening one’s 
environmental impact or carbon footprint. They offer products and services that are environmentally friendly and 
sustainable, in line with their values and ideals. This model can help these companies achieve their goals by 
providing access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - 
thus increasing their insights and informing future marketing strategies as we would show.
			  </h4>
				
				"""

			st.markdown(text1, unsafe_allow_html=True)

		with col2:

			text2 = """
				
<h3>Data Details</h3>				
<h4 style="text-align:justify; line-height:150%; color: black; ">	
					  
The quality of a model is only as good as the quality of data used to generate it. 
To produce our models, we must first select an appropriate set of data for training 
that has been annotated. The data collection used for the development of this model 
was funded by the Canadian Foundation for Innovation JELF Grant to Chris Bauch, University 
of Waterloo. The dataset aggregates tweets pertaining to climate change collected between 
Apr 27, 2015, and Feb 21, 2018. In total, 43943 tweets. Each tweet is labeled as one of the following classes:

<h3>Class Description</h3>
<ul type="A" style="font-weight: bold;">
  <li><span>2 NEWS:* the tweet links to factual news about climate change</span></li>
  <li><span>1 PRO: the tweet supports the belief of man-made climate change</span></li>
  <li><span>0 NEUTRAL: the tweet neither supports nor refutes the belief of man-made climate change</span></li>
  <li><span> -1 ANTI: the tweet does not believe in man-made climate change</span></li>
  
</ul>	


<h3>Variable definitions</h3>
<ul type="A" style="font-weight: bold;">
  <li><span>sentiment: Sentiment of a tweet</span></li>
  <li><span>message: Tweet body</span></li>
  <li><span>sentiment: Sentiment of a tweet</span></li>
  
</ul>	
 				
</h4>
					"""
			st.markdown(text2, unsafe_allow_html=True)

		with col3:
			#st.header("An owl")
			text3 = """

<h3>Model goals</h3> 
<h4 style="text-align:justify; line-height:130%; color: black;">

This model aims to explore machine learning as a method to assist us in identifying whether or not a person 
believes in climate change and ascertain if such a person could be converted to a new customer for an 
environmentally conservative product based on their tweets. Our  ML models are capable of classifying 
tweets into any of 4 categories as detailed below. </h4> 
 
<h3>Model performance</h3/>  
<h4 style="text-align:justify; line-height:130%; color: black;">

Our model performance is gauged using the F1 score. 
This Score is the weighted average of Precision and Recall. 
The F1 score is the industry standard for measuring the accuracy of a classification model as it takes both 
false positives and false negatives into account.  
 F1 is usually more useful than accuracy, especially if you have an uneven class distribution. 
The F1 score for the models we deployed here ranges from 0.76 to 0.88. 
 
 </h4
 						"""

			st.markdown(text3, unsafe_allow_html=True)

			st.info("General Information about the Data")
			# You can read a markdown file from supporting resources folder
			st.markdown("Some information here")

			st.header("Raw Twitter data and label")
			if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']])  # will write the df to the page


	if selected == "Contact Us":
			image = Image.open('IMG-20220629-WA0005.jpg')
			st.image(image, width=150)
			st.title("Contact us")
			st.write("If you have any comments or questions about this app or Data Fluent Inc. in general, feel free to contact us.")

			st.text_input('Email address')
			st.text_area("Enter your message")
			st.button("Send")




# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
