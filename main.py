#importing libraries
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import plotly.express as px 
from pywaffle import Waffle 
import plotly.graph_objects as go 
from wordcloud import WordCloud
import mplcursors 
import matplotlib.cm as cm
#import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
#from tensorflow.keras.layers import Embedding


plt.style.use("ggplot")
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data():
    data = pd.read_csv("reccomendation_df3.csv")       
    return data

df = load_data()


#df = pd.read_csv("FinalDataset1.csv")
#Navigation slider
rad=st.sidebar.radio("NAVIGATION BAR",["Home","Insights","Recommendation System","Error Handling"])

if rad == "Home":
    
    #project Title
    st.title(":blue[INDIAN CUISINE ANALYSIS AND RECOMMENDATION SYSTEM ]")

    st.image("cuisine.jpeg",width=700) 
    st.dataframe(df,width=800,height=500)



elif rad == "Insights":

    charts = ["Most used ingredients in India","Proportion of flavor profiles","Course Meal with shortest cooking time",
              "Ingredients used in different meals","Region wise Veg and Non-Veg cuisine",
              "Daily Meals with shortest time(prep+cook)", "Ingredients used in Diet based food",
              "Region wise distribution of Flavors","Famous international cuisines","Region wise course distribution"]
    

    select=st.sidebar.selectbox("select the insight",charts)

    if select == charts[0]:
        st.write(""" ## Most Used Ingredients In Indian Cuisines""")
        ingredientCharts = ["Pie","Sunburst"]
        select=st.selectbox("Select the Chart",ingredientCharts)
        if select == ingredientCharts[0]:

            splitted_lst = df['ProcessedCleanedLoweredIngredientsFiltered'].apply(lambda x: x.split(','))

            #top=st.text_input("Enter No. of ingredients you want to see")
         
            # Flatten the list of lists in the 'Ingredients' column
            flat_list = [word for sublist in splitted_lst for word in sublist]

            # Count the occurrences of each word
            word_counts = Counter(flat_list)
            top_common_words1 = word_counts.most_common(10)

            top_words_df = pd.DataFrame(top_common_words1, columns=['Word', 'COUNTS'])
            #top_words_df

            fig = px.pie(top_words_df, names='Word', values='COUNTS')
            fig.update_layout(width=800, height=600)

            #fig.show()
            st.plotly_chart(fig)

        else:
            # Your code to count the occurrences of each word
            splitted_lst = df['ProcessedCleanedLoweredIngredientsFiltered'].apply(lambda x: x.split(','))
            flat_list = [word for sublist in splitted_lst for word in sublist]
            word_counts = Counter(flat_list)
            top_common_words1 = word_counts.most_common(10)

            # Convert top_common_words1 to a DataFrame
            top_words_df = pd.DataFrame(top_common_words1, columns=['Word', 'COUNTS'])

            # Create a sunburst plot
            fig = px.sunburst(top_words_df, path=['Word'], values='COUNTS', color='COUNTS',color_continuous_scale='Plasma')
            fig.update_layout(width=800, height=600)

            # Show the plot
            #fig.show()
            st.plotly_chart(fig)



    elif select == "Proportion of flavor profiles":
        st.write(""" ## Proportion of flavor profiles""")

        flavoured_profile_index = list(df["FlavourProfile"].value_counts().index)
        flavoured_profile_count = list(df["FlavourProfile"].value_counts())

        flavorChart = ["Waffle Plot", "Pie Chart", "Scatter Plot"]
        select=st.selectbox("Select the chart",flavorChart)

        if select == flavorChart[0]:
            data = dict(df["FlavourProfile"].value_counts(normalize=True) * 100)
            flavors = df["FlavourProfile"]

            fig = plt.figure(
                FigureClass=Waffle,
                rows=10,  # Rows represent the total number of waffles
                columns=15,  # Columns are not used for this type of chart
                values=data,
                title={'label': 'Proportion of Flavor Profiles', 'loc': 'center', 'fontsize': 15},
                facecolor='black',
                labels=[f"{k} ({v:.2f}%)" for k, v in data.items()],
                legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1), 'ncol':1, 'framealpha': 0, 'prop': {'size': 12}  },
                figsize=(9,20)
            )
          
            legend = fig.gca().get_legend()
            for text in legend.get_texts():
                text.set_color('white')
            #plt.show()
            st.pyplot()

        elif select == flavorChart[1]:
            labels = flavoured_profile_index
            values = flavoured_profile_count

            fig = go.Figure(data=[go.Pie(labels=labels, values=values)] )
            #fig.show()
            fig.update_layout(width=800, height=600)

            st.plotly_chart(fig)

        else:
            x = flavoured_profile_index
            y = flavoured_profile_count

            
            # plotting scatter plot 
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 
                    size = 15,
                    color=np.random.randn(10), 
                    colorscale='Viridis',  
                    showscale=True
                    ) ,
                    text=flavoured_profile_index)) 
            fig.update_layout(
                #title="Flavoured Profile Scatter Plot"
                width=800,
                height=800
            )   
            st.plotly_chart(fig)

       
       


    elif select == charts[2]:
        st.write(""" ## Top N Courses With Shortest Cooking Time""")

        dishchart = ["Main Course", "Side Dish", "Dessert"]
        select=st.selectbox("Select the Course Meal ",dishchart)
        
        if select == dishchart[0]:
            #num=st.text_input("Enter No. of recipes")
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            #main_courses_shortest['Color'] = dataset['TotalTimeInMins'].map(cooking_time_to_color)
            main_courses_shortest = df[df['Course'] == 'Main Course'].sort_values(by='TotalTimeInMins').head(slide)
            #print(main_courses_shortest)
            #cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            fig = px.bar(main_courses_shortest, x='TotalTimeInMins', y='Course', 
                        #title='Top 10 Main Courses with Shortest Cooking Time',
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=main_courses_shortest["EnglishRecepie"],
                        #color='Color')  # Color bars based on cooking time
                        color='TotalTimeInMins',
                        color_continuous_scale='viridis_r')
            fig.update_layout(width=1000, height=800) 

            #fig.show()
            st.plotly_chart(fig)

        elif select == dishchart[1]:
            #cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            #main_courses_shortest['Color'] = dataset['TotalTimeInMins'].map(cooking_time_to_color)
            #num=int(st.text_input("Enter No. of recipes"))
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            main_courses_shortest = df[(df['Course'] == 'Side Dish') & (df['TotalTimeInMins'] > 0)].sort_values(by='TotalTimeInMins').head(slide)
            #print(main_courses_shortest)
            #cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            fig = px.bar(main_courses_shortest, x='TotalTimeInMins', y='Course', 
                        #title='Top 10 side dishes with shortest cooking time',
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=main_courses_shortest["EnglishRecepie"],
                        #color='Color')  # Color bars based on cooking time
                        color='TotalTimeInMins',
                        color_continuous_scale='viridis_r')
            fig.update_layout(width=1000, height=800) 

            #fig.show()
            st.plotly_chart(fig)

        else:
            #cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            #main_courses_shortest['Color'] = dataset['TotalTimeInMins'].map(cooking_time_to_color)
            #num=int(st.text_input("Enter No. of recipes"))
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            main_courses_shortest = df[(df['Course'] == 'Dessert') & (df['TotalTimeInMins'] > 0)].sort_values(by='TotalTimeInMins').head(slide)
            #print(main_courses_shortest)
            #cooking_time_to_color = {10: 'red', 15: 'green', 20: 'blue'} 
            fig = px.bar(main_courses_shortest, x='TotalTimeInMins', y='Course', 
                        #title='Top 10 Dessert with shortest cooking time',
                        category_orders={"CookTimeInMins": [10, 15, 20]},
                        text=main_courses_shortest["EnglishRecepie"],
                        #color='Color')  # Color bars based on cooking time
                        color='TotalTimeInMins',
                        color_continuous_scale='jet')
            fig.update_layout(width=1000, height=800) 
            st.plotly_chart(fig)

    elif select == charts[3]:
        st.write(""" ## Top Ingredients In Different Meals""")

        dishchart = ["Main Course", "Side Dish", "Dessert"]
        select=st.selectbox("Select the Course Meal ",dishchart)

        if select == dishchart[0]:
            # Sample dessert ingredient data (replace with your own data)
            dessert_df = df[df['Course'] == 'Main Course'].reset_index()
            ingredients = []
            for i in range(0, len(dessert_df)):
                text = dessert_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='prism',  # Use the 'plasma' colormap
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud_plasma)
            #plt.title('Word Cloud (Plasma Colormap)')
            plt.axis('off')

            plt.tight_layout()
            #plt.show()
            st.pyplot()

        elif select == dishchart[1]:
            side_dish_df = df[df['Course'] == 'Side Dish'].reset_index()
            ingredients = []
            for i in range(0, len(side_dish_df)):
                text = side_dish_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='plasma',  # Use the 'plasma' colormap
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud_plasma)
            #plt.title('Word Cloud (Plasma Colormap)')
            plt.axis('off')

            plt.tight_layout()
            #plt.show()
            st.pyplot()

        else:
            dessert_df = df[df['Course'] == 'Dessert'].reset_index()
            ingredients = []
            for i in range(0, len(dessert_df)):
                text = dessert_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            # Generate the word cloud using the 'plasma' colormap
            wordcloud_plasma = WordCloud(
                width=400,
                height=400,
                colormap='coolwarm',  # Use the 'plasma' colormap
                background_color='black',
                min_font_size=8
            ).generate(text)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud_plasma)
            #plt.title('Word Cloud (Plasma Colormap)')
            plt.axis('off')

            plt.tight_layout()
            #plt.show()
            st.pyplot()

    elif select == charts[4]:
      st.write(""" ## Distribution of Veg/Non-Veg Dishes By Region""")
      vegnonveg = ["Vegeterian", "Non-Vegeterian"]
      select = st.selectbox("Select the meal of the day",vegnonveg)

      if select == vegnonveg[0]:
            vegetarian_df = df[df['Diet'] == 'Vegetarian']

            # Create a DataFrame that groups data by 'Region' and calculates counts for vegetarian dishes
            vegetarian_counts = vegetarian_df['Region'].value_counts()

            # Create a bar chart for vegetarian dishes
            fig, ax = plt.subplots(figsize=(17, 10))
           # bars = ax.bar(x_values, vegetarian_counts, color='green')
        
            bars = vegetarian_counts.plot(kind='bar', color='green', ax=ax)
            fig.patch.set_facecolor('none')  # Set the figure's background color to be transparent
            ax.set_facecolor('none')        
            ax.grid(False)
            # Add counts inside the bars
            ax.set_xlabel("Region",color='white',fontsize=16)
            ax.set_ylabel("Number of Vegetarian Dishes",color='white',fontsize=16)
            #ax.set_title("Distribution of Vegetarian Dishes by Region")
            ax.tick_params(axis='x', colors='white',labelsize=15)  # Change the color of tick labels on the x-axis to red
            ax.tick_params(axis='y', colors='white',labelsize=15) 
            #plt.show()
            st.pyplot()

      else:
            non_vegetarian_df = df[df['Diet'] == 'Non Vegeterian']

            # Create a DataFrame that groups data by 'Region' and calculates counts for non-vegetarian dishes
            non_vegetarian_counts = non_vegetarian_df['Region'].value_counts()

            # Create a bar chart for non-vegetarian dishes
            fig, ax = plt.subplots(figsize=(17, 10))
            non_vegetarian_counts.plot(kind='bar', color='red', ax=ax)
            fig.patch.set_facecolor('none')  # Set the figure's background color to be transparent
            ax.set_facecolor('none') 
            ax.set_xlabel("Region",color='white',fontsize=13)
            ax.set_ylabel("Number of Non-Vegetarian Dishes",color='white',fontsize=13)
            ax.grid(False)

            #ax.set_title("Distribution of Non-Vegetarian Dishes by Region")
            ax.tick_params(axis='x', colors='white',labelsize=12)  # Change the color of tick labels on the x-axis to red
            ax.tick_params(axis='y', colors='white',labelsize=12) 
            #plt.show()
            st.pyplot()

    elif select ==  charts[5]:
        st.write(""" ## Daily Meals With Shortest Time(Prep+Cook) """)

        dishchart = ["Breakfast", "Lunch", "Dinner"]
        select = st.selectbox("Select the meal of the day",dishchart)

        if select == dishchart[0]:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            # Filter data for Breakfast
            filtered_df = df[df['Course'].isin(['Indian Breakfast'])]

            # Find the recipe with the least total time
            min_time_recipe = filtered_df[filtered_df['TotalTimeInMins'] == filtered_df['TotalTimeInMins']]

            # Display the recipe with the least time required
            print("Recipe with the least time required for Breakfast:")

            Breakfast = min_time_recipe.sort_values('TotalTimeInMins').head(slide)
            # Sort the data based on the 'TotalTimeInMins' column
            Breakfast_sorted = Breakfast.sort_values(by='TotalTimeInMins')

            # Define a colormap based on the 'TotalTimeInMins' values
            colors = plt.cm.jet(Breakfast_sorted['TotalTimeInMins'] / Breakfast_sorted['TotalTimeInMins'].max())
            
            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Breakfast_sorted['EnglishRecepie'], Breakfast_sorted['TotalTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            #plt.title('Recipes with the Least Total Time for Dinner')
            plt.xticks(rotation=45,color='white',fontsize=12)
            plt.yticks(color='white',fontsize=12)

            plt.gca().set_facecolor('black')
            # Set xticks explicitly with sorted recipe names
            plt.xticks(Breakfast_sorted['EnglishRecepie'], rotation=45, ha='right')
            plt.gca().grid(False)
            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Breakfast_sorted['TotalTimeInMins'].min(), vmax=Breakfast_sorted['TotalTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total Time (mins)')

            plt.tight_layout()

            # Display the chart
            #plt.show()

            st.pyplot()


        elif select == dishchart[1]:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)


            # Filter data for Lunch
            filtered_df = df[df['Course'].isin(['Lunch']) & (df['TotalTimeInMins'] > 0)]

            # Find the recipe with the least total time
            min_time_recipe = filtered_df[filtered_df['TotalTimeInMins'] == filtered_df['TotalTimeInMins']]

            # Display the recipe with the least time required
            print("Recipe with the least time required for Lunch or Dinner:")

            Lunch = min_time_recipe.sort_values('TotalTimeInMins').head(slide)
            Lunch_sorted = Lunch.sort_values(by='TotalTimeInMins')

            # Define a colormap based on the 'TotalTimeInMins' values
            colors = plt.cm.jet(Lunch_sorted['TotalTimeInMins'] / Lunch_sorted['TotalTimeInMins'].max())

            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Lunch_sorted['EnglishRecepie'], Lunch_sorted['TotalTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            plt.title('Recipes with the Least Total Time for Lunch')
            plt.xticks(rotation=45,color='white',fontsize=12,ha='right')
            plt.yticks(color='white',fontsize=12)
            plt.gca().grid(False)
            plt.gca().set_facecolor('black')

            # Set the x-axis ticks and labels explicitly
            plt.xticks(range(len(Lunch_sorted)), Lunch_sorted['EnglishRecepie'])

            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Lunch_sorted['TotalTimeInMins'].min(), vmax=Lunch_sorted['TotalTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total Time (mins)')

            plt.tight_layout()

            # Display the chart
            #plt.show()

            st.pyplot()


        else:
            slide = st.slider("Number of courses to show",min_value=2,max_value=15,value=10)

            # Filter data for Dinner
            filtered_df = df[df['Course'].isin(['Dinner'])]

            # Find the recipe with the least total time
            min_time_recipe = filtered_df[filtered_df['TotalTimeInMins'] == filtered_df['TotalTimeInMins']]

            # Display the recipe with the least time required
            print("Recipe with the least time required for Dinner:")

            Dinner = min_time_recipe.sort_values('TotalTimeInMins').head(slide)
            # Sort the data based on the 'TotalTimeInMins' column
            Dinner_sorted = Dinner.sort_values(by='TotalTimeInMins')

            # Define a colormap based on the 'TotalTimeInMins' values
            colors = plt.cm.jet(Dinner_sorted['TotalTimeInMins'] / Dinner_sorted['TotalTimeInMins'].max())

            # Create a bar chart with sorted and colored data
            plt.figure(figsize=(15, 10),facecolor='black')
            bars = plt.bar(Dinner_sorted['EnglishRecepie'], Dinner_sorted['TotalTimeInMins'], color=colors)
            plt.xlabel('Recepie',color = 'white',fontsize=13)
            plt.ylabel('Total Time (mins)',color='white',fontsize=13)
            #plt.title('Recipes with the Least Total Time for Dinner')
            plt.xticks(rotation=45,color='white',fontsize=12,ha='right')
            plt.yticks(color='white',fontsize=12)
            plt.gca().grid(False)
            plt.gca().set_facecolor('black')
            # Set xticks explicitly with sorted recipe names
            plt.xticks(Dinner_sorted['EnglishRecepie'], rotation=45, ha='right')

            # Add a colorbar to represent the mapping of colors to time values
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=Dinner_sorted['TotalTimeInMins'].min(), vmax=Dinner_sorted['TotalTimeInMins'].max()))
            sm.set_array([])  # Dummy array for the colorbar
            cbar = plt.colorbar(sm)
            cbar.set_label('Total Time (mins)')

            plt.tight_layout()

            # Display the chart
            #plt.show()

            st.pyplot()

    elif select == charts[6]:
        st.write(""" ## Most Common Ingredients In Diet Based Food""")
        diet = ["Diabetic friendly", "High protien veg", "High protien non-veg", "Vegan"]
        select = st.selectbox("Select the type of diet",diet)

        if select == diet[0]:
            diabetic_Friendly_df  = df[df['Diet'] == 'Diabetic Friendly'].reset_index()

            ingredients = []
            for i in range(0,len(diabetic_Friendly_df)):
                text = diabetic_Friendly_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'plasma'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            #plt.show()
            st.pyplot()

        elif select == diet[1]:
            vegeterian_df  = df[df['Diet'] == 'High Protein Vegetarian'].reset_index()

            ingredients = []
            for i in range(0,len(vegeterian_df)):
                text = vegeterian_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'prism'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            #plt.show()
            st.pyplot()


        elif select == diet[2]:
            non_vegeterian_df  = df[df['Diet'] == 'High Protein Non Vegetarian'].reset_index()

            ingredients = []
            for i in range(0,len(non_vegeterian_df)):
                text = non_vegeterian_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'coolwarm'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            #plt.show()

            st.pyplot()

        elif select == diet[3]:

            vegan_df  = df[df['Diet'] == 'Vegan'].reset_index()

            ingredients = []
            for i in range(0,len(vegan_df)):
                text = vegan_df["ProcessedCleanedLoweredIngredientsFiltered"][i].split(',')
                text = ','.join(text)
                ingredients.append(text)
                text = ' '.join(ingredients)

            wordcloud = WordCloud(width = 400, height = 400, colormap = 'gnuplot2'
                                ,background_color ='black', 
                            min_font_size = 8).generate(text)                  
            plt.figure(figsize = (10, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis('off')
            #plt.show()
            st.pyplot()

    elif select == charts[7]:
        st.write("""## Distribution Of Flavors Region Wise """)
        flavorChart = ["Sweet","Spicy","Tangy","Bitter","Savory"]
        select = st.selectbox("Select the flavor",flavorChart)

        if select == flavorChart[0]:
            # Filter for recipes with 'Sweet' FlavourProfile
            sweet_recipes = df[df['FlavourProfile'] == 'Sweet']

            # Group sweet recipes by 'Region' and count the occurrences
            region_wide_sweet_distribution = sweet_recipes['Region'].value_counts()
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Sweet' FlavourProfile
            sweet_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Sweet']

            # Group sweet recipes by 'Region' and count the occurrences
            region_wide_sweet_distribution = sweet_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['blue', 'green', 'orange', 'red', 'purple'] 

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_sweet_distribution, x=region_wide_sweet_distribution.index, y='Region',
                        color=region_wide_sweet_distribution.index, color_discrete_sequence=colors,
                        #title='Distribution of Region Wise Sweet Recipes '
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Sweet Recipes')

            # Show the interactive chart
            #fig.show()
            st.plotly_chart(fig)

        elif select == flavorChart[1]:
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Spicy' FlavourProfile
            spicy_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Spicy']

            # Group spicy recipes by 'Region' and count the occurrences
            region_wide_spicy_distribution = spicy_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['red', 'orange', 'yellow', 'purple', 'blue']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_spicy_distribution, x=region_wide_spicy_distribution.index, y='Region',
                        color=region_wide_spicy_distribution.index, color_discrete_sequence=colors,
                        #title='Distribution of "Spicy" Recipes by Region '
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Spicy Recipes')

            # Show the interactive chart
            #fig.show()
            st.plotly_chart(fig)


        elif select  == flavorChart[2]:
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Tangy' FlavourProfile
            tangy_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Tangy']

            # Group tangy recipes by 'Region' and count the occurrences
            region_wide_tangy_distribution = tangy_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['orange', 'yellow', 'green', 'red', 'purple']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_tangy_distribution, x=region_wide_tangy_distribution.index, y='Region',
                        color=region_wide_tangy_distribution.index, color_discrete_sequence=colors,
                        #title='Distribution of "Tangy" Recipes by Region '
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Tangy Recipes')

            # Show the interactive chart
            #fig.show()
            st.plotly_chart(fig)

        elif select == flavorChart[3]:

            
            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Bitter' FlavourProfile
            bitter_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Bitter']

            # Group bitter recipes by 'Region' and count the occurrences
            region_wide_bitter_distribution = bitter_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['red', 'orange', 'yellow', 'purple', 'blue']

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_bitter_distribution, x=region_wide_bitter_distribution.index, y='Region',
                        color=region_wide_bitter_distribution.index, color_discrete_sequence=colors,
                        #title='Distribution of "Bitter" Recipes by Region '
                         )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Bitter Recipes')

            # Show the interactive chart
            #fig.show()
            st.plotly_chart(fig)


        else:
            

            # Filter out "International" and "Indian" regions
            filtered_df = df[~df['Region'].isin(['International', 'Indian'])]

            # Filter for recipes with 'Savory' FlavourProfile
            savory_recipes = filtered_df[filtered_df['FlavourProfile'] == 'Savory']

            # Group savory recipes by 'Region' and count the occurrences
            region_wide_savory_distribution = savory_recipes['Region'].value_counts()

            # Define a list of colors for each region
            colors = ['green', 'brown', 'olive', 'darkgreen', 'darkolivegreen']  # Add more colors as needed

            # Create an interactive bar chart using Plotly with different colors
            fig = px.bar(region_wide_savory_distribution, x=region_wide_savory_distribution.index, y='Region',
                        color=region_wide_savory_distribution.index, color_discrete_sequence=colors,
                        #title='Distribution of "Savory" Recipes '
                        )
            fig.update_xaxes(title='Region')
            fig.update_yaxes(title='Number of Savory Recipes')

            # Show the interactive chart
            #fig.show()
            st.plotly_chart(fig)

    elif select == charts[8]:
        st.write(""" ## Famous International Cuisines In India """)
         # List of cuisines that are not of Indian origin
        international_cuisines = ['Thai', 'Continental','Mexican', 'Italian Recipes','Chinese','Middle Eastern', 'European','Arab','Japanese','Vietnamese', 'British','Greek', 'Nepalese','French',  'Mediterranean', 'Sri Lankan', 'Indonesian', 'African', 'Korean', 'American', 'Pakistani', 'Caribbean','World Breakfast', 'Malaysian','Jewish', 'Burmese', 'Afghan']

        # Filter the DataFrame to include only international cuisines famous in India
        famous_international_cuisines = df[df['Cuisine'].isin(international_cuisines)]


        famous_international_cuisines = pd.DataFrame(famous_international_cuisines)
        famous_international_cuisines.sample(5)
        # Count the occurrences of each cuisine
        cuisine_counts = famous_international_cuisines['Cuisine'].value_counts()
        # Get the top 25 most popular cuisines
        top_cuisines = cuisine_counts.head(10)

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(top_cuisines, names=top_cuisines.index, values=top_cuisines.values,
                    title='Top 10 Most Popular Cuisines', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set1)

        # Update layout for better readability
        fig.update_layout(
            title_text='Top 10 Most Popular Cuisines',
            showlegend=True,
            legend=dict(title='Cuisine'),
            margin=dict(t=0, b=0, l=0, r=0)
        )

        # Display the interactive plot
        #fig.show()
        st.plotly_chart(fig)

    elif select == charts[9]:
        st.write(""" ## Region Wise Course Distribution""")
                
        
        regions_to_exclude = ['International', 'Other', 'Indian']

        # Filter the DataFrame to exclude specified regions
        filtered_df = df[~df['Region'].isin(regions_to_exclude)]

        # Define the courses you want to count
        courses_to_include = ['Dinner', 'Lunch', 'Indian Breakfast', 'Snack', 'Brunch']

        # Filter the DataFrame to include only specified courses
        filtered_df = filtered_df[filtered_df['Course'].isin(courses_to_include)]

        # Group the data by region and count the number of recipes in each region
        region_counts = filtered_df['Region'].value_counts()

        # Create a dictionary to store the counts of courses by region
        course_counts_by_region = {}

        # Populate the dictionary with zeros for each course in each region
        for region in filtered_df['Region'].unique():
            course_counts_by_region[region] = {course: 0 for course in courses_to_include}

        # Simulate course counts for each region (replace with your actual data)
        for region in filtered_df['Region'].unique():
            for course in courses_to_include:
                course_counts_by_region[region][course] = random.randint(5, 20)

        # Create a DataFrame from the course counts by region dictionary
        course_counts_df = pd.DataFrame(course_counts_by_region).T

        # Plotting the grouped bar chart with increased width of bars
        fig, ax = plt.subplots(figsize=(10, 6))

        # Specify the width of the bars using the 'width' parameter
        bar_width = 0.7
        course_counts_df.plot(kind='bar', ax=ax, width=bar_width)

        plt.xlabel("Region")
        plt.ylabel("Number of Recipes")
        #plt.title("Course-Wise Distribution of Regions")

        plt.legend(title="Courses", loc="upper right")

        # Set the y-axis to display integer values
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()

        st.pyplot()

#Recommendation system
elif rad == "Recommendation System":
    st.write(""" ## Recommedation System for Recipes using Ingredients  """)
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)
    print("Model Loaded")


    #@st.cache_data
    #def load_data():
        #data = pd.read_csv("reccomendation_df.csv")       
        #return data

    #data = load_data()
    Recepies = df[['Srno','EnglishRecepie','StringCleanedLowerIngredientsFiltered']]
    Recepies.dropna(inplace=True)
    Recepies = Recepies.rename(columns={'StringCleanedLowerIngredientsFiltered':'Tags'})

    titles = list(Recepies['Tags'])
    def embed(texts):
        return model(texts)
    
    #embed(['This is good Recipie'])
    embeddings = embed(titles)
    num=int(st.selectbox("Number of Recipes",[1,2,3,4,5,6,7,8,9,10],index=0))

    #pca = PCA(n_components=2)
    #emd_2d = pca.fit_transform(embeddings)
    nn = NearestNeighbors(n_neighbors=num)
    nn.fit(embeddings)
    def recommend(text):
        emb = embed([text])
        neighbours = nn.kneighbors(emb, return_distance=False)[0]
        
        similar_recipes = Recepies.iloc[neighbours]
        sorted_recipes = similar_recipes.sort_values(by='EnglishRecepie', ascending=False)
        recipe = sorted_recipes['EnglishRecepie'].to_list()
       
        #recipe_names = [recipe.split(': ')[1] for recipe in final_list]

        return recipe
    
    text=st.text_input("Enter the ingredient(s) space separated")

    def evaluate_recommendations(recommendation, user_ingredients):
        num_correct = 0

        for recipe_name in recommendation:
            recipe_row = data[data["EnglishRecepie"] == recipe_name].iloc[0]
            recommended_ingredients = recipe_row["StringCleanedLowerIngredientsFiltered"]
            
            if all(ingredient in recommended_ingredients for ingredient in user_ingredients):
                num_correct += 1

        accuracy = num_correct / len(recommendation)
        return accuracy

    final_list=[]
    accuracy = 0.00
    if text:
        final_list=recommend(text)
        # Calculate accuracy  
        accuracy = evaluate_recommendations(final_list, text)
        #st.balloons()
        if accuracy > 0.00:
            for index, recipe in enumerate(final_list, start=1):
                st.write(index, f" <span style='color:#fa8072'>{recipe}</span>", unsafe_allow_html=True)
        else:
            st.error("Please enter valid ingredients")
        
        # Print the accuracy
        st.write(f"Accuracy: {accuracy:.2%}") 
        
    else:
        st.error("Please enter ingredient(s) as well")   



   

else:
    pass

