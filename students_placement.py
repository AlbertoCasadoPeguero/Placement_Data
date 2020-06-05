#Importing the libraries
import numpy as np
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Placement_Data.csv')

#Changing columns names
dataset.columns = ['Serial','Gender','Secondary_Edu','Board_Edu','Higher_Second_Edu',
                   'Board_Edu_2','Specialization_Higher_Second_Edu','Degree',
                   'Under_Grad_Degree','Experience','Employability_Test',
                   'Post_Grad_Specialization','MBA','Placement','Salary']

#EDA