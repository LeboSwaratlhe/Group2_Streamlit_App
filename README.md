# Anime Recommender system team 2 - Streamlit App

<div id="s_image" align="center">
  <img src="Streamlit app screenshot.jpeg" width="850" height="400" alt=""/>
</div>
---

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Team Members](#team-members)
---

## About the Project <a class="anchor" id="about-the-project"></a> 

This project focuses on the development of a hybrid recommender system that integrates collaborative filtering and content-based approaches for a dataset of anime titles. The system is designed to accurately predict user ratings for anime titles that have not yet been viewed, leveraging the user's historical preferences and behavioral data to generate personalized recommendations

---

## Features <a class="anchor" id="features"></a>
- **Responsive Design**: Works on both desktop and mobile devices.
- **Interactive Elements**: Buttons, expandable project sections, and a contact form.
- **Easy to Customize**: Modify the content to suit your needs.

---

## Installation <a class="anchor" id="installation"></a>
To run this app locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SaneleBhembe/Group2_Streamlit_App.git
   cd Group2_Streamlit_App
2. **Set Up a Virtual Environment (Optional but Recommended)**
   ```bash
   ``python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install Dependencies**
   ```bash
      pip install -r requirements.txt

4. **Run the Streamlit App**
   ```bash
      streamlit run app.py
---

## Usage <a class="anchor" id="usage"></a>
Prerequisites
- `Python 3.7 or higher`
- `pip (Python package manager)`
---
## Deployment <a class="anchor" id="deployment"></a>

The Anime Recommendation System is a user-friendly application designed to provide personalized anime suggestions based on advanced machine learning techniques. Below is an overview of the application interface and its functionality:

#### **Application Interface**

**Header:**  
**Anime Recommendation System**  
*Recommendation | Team*  

**Main Functionality:**  

1. **Select Recommendation Method**  
   Users can choose from the following recommendation methods:  
   - **1. Content-Based (PCA):** Utilizes Principal Component Analysis to recommend anime titles based on content similarity.  
   - **2. Collaborative-Based (Rating Predictor - NCF):** Employs Neural Collaborative Filtering to predict user ratings for unseen anime titles.  
   - **3. Collaborative-Based (User Recommendations):** Generates recommendations based on similar users' preferences.  
   - **4. Hybrid (PCA + NCF):** Combines content-based and collaborative filtering approaches for enhanced recommendation accuracy.  

2. **Select an Anime Title:**  
   Users can input or select an anime title (e.g., *"Kimi no Na Wa."*) to receive personalized recommendations.  


#### **How It Works**  
1. The user selects a recommendation method based on their preference.  
2. The user inputs or selects an anime title they enjoy.  
3. The system processes the input using the chosen method and generates a list of recommended anime titles tailored to the user's preferences.  


#### **Deployment Details**  
- **Backend:** The system is powered by machine learning models (PCA, NCF, and hybrid techniques) deployed on a scalable cloud platform.  
- **Frontend:** A clean and intuitive interface ensures seamless user interaction.  
- **Integration:** The app is integrated with a database of anime titles and user ratings to ensure accurate and real-time recommendations.  


This deployment provides a robust and interactive platform for anime enthusiasts to discover new titles aligned with their tastes. Let us know if you need further details or assistance!


---
## Technologies Used <a class="anchor" id="technologies-used"></a>
- `Streamlit: For building and deploying the web app`
- `Python: The primary programming language`
- `Pandas: For data manipulation and analysis`
- `NumPy: For numerical computations`
- `Scikit-learn: For machine learning`
- `Dill`
- `MLFlow, Pickle and surprise `


---
## Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
|[Nombulelo Perfidia Tracy Nyoni]                                                             |	nombulelotracy@gmail.com
|[Phillip	Sethole]                                                                            |	philipsethole346@gmail.com
|[Sanele Bhembe] 	                                                                            | sanelebhembe12@gmail.com
|[Fransisca Matlou Nong]	                                                                    | matlou9637@gmail.com
|[Mzwandile Stuurman]	                                                                        | stuurmanmzwandile@gmail.com
|[Lebogang Swaratlhe]	                                                                        | lebogangswaratlhe@gmail.com

