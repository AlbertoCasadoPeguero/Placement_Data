#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns

#Import the dataset
dataset = pd.read_csv('Placement_Data.csv')

#Changing columns names
dataset.rename(columns = {'sl_no':'serial','ssc_p':'secondary_edu','ssc_b':'board_edu',
                'hsc_p':'higher_second_edu','hsc_b':'board_edu_2','hsc_s':'second_edu_speci',
                'degree_p':'degree','degree_t':'degree_title','workex':'experience',
                'etest_p':'employability_test','specialisation':'MBA_title',
                'mba_p':'MBA','status':'placement'},inplace = True)

#EDA
dataset.info()

#The data is a little unbalanced
sns.countplot('placement',data = dataset)

sns.countplot('placement',hue = 'gender',data = dataset)
sns.countplot('placement',hue = 'board_edu',data = dataset)
sns.countplot('placement',hue = 'board_edu_2',data = dataset)
sns.countplot('placement',hue = 'second_edu_speci',data = dataset)
sns.countplot('placement',hue = 'degree_title',data = dataset)
sns.countplot('placement',hue = 'experience',data = dataset)

no_exp_degree_not_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Not Placed')]['degree_title']
no_exp_degree_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Placed')]['degree_title']
no_exp_board_not_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Not Placed')]['board_edu']
no_exp_board_placed = dataset[(dataset['experience'] == 'No')&(dataset['placement'] == 'Placed')]['board_edu']

sns.countplot(no_exp_degree_not_placed)
sns.countplot(no_exp_degree_placed)
sns.countplot(no_exp_board_not_placed)
sns.countplot(no_exp_board_placed)

sns.countplot('placement',hue = 'MBA_title',data = dataset)

sns.scatterplot(range(0,len(dataset['secondary_edu'])),'secondary_edu',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['higher_second_edu'])),'higher_second_edu',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['degree'])),'degree',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['employability_test'])),'employability_test',
                hue = 'placement',data = dataset)
sns.scatterplot(range(0,len(dataset['MBA'])),'MBA',
                hue = 'placement',data = dataset)

#Since the salary has nan values is not a good choice to make a predictor on the salary
#Since it will to bias towards the placed ones so it became a classificacion problem