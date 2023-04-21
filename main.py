# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:00:46 2023

@author: 91836
"""


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

app = FastAPI()

class model_input(BaseModel):
    
    age : int
    gender     : str
    self_employed  : str
    family_history : str
    work_interfere  : str
    no_employees     : str
    remote_work       : str
    tech_company      : str
    benefits          : str
    care_options        : str
    wellness_program   : str
    seek_help          : str
    anonymity          : str
    leave              : str
    mental_health_consequence : str
    phys_health_consequence    : str
    coworkers          :   str
    supervisor   :            str
    mental_health_interview    : str
    phys_health_interview    : str
    mental_vs_physical      : str
    obs_consequence     : str
        
# loading the saved model
loaded_model = pickle.load(open('prediction_model.pkl', 'rb'))

@app.post('/mental_health_prediction')
def mentalHealth_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    
    
    age = input_dictionary['age']
    
    gender = input_dictionary['gender']
    
    if( gender=='Male'):
        gender= 0
    else:
        gender= 1
    
    self_employed = input_dictionary['self_employed']
    
    if( self_employed=='No'):
        self_employed= 0
    else:
        self_employed= 1
        
    family_history = input_dictionary['family_history']
    if(family_history =='No'):
        family_history= 0
    else:
        family_history= 1
                    
    work_interfere = input_dictionary['work_interfere']
    if(work_interfere =='Never'):
        work_interfere= 0
    elif(work_interfere =='Rarely'):
        work_interfere= 1
    elif(work_interfere =='Sometimes'):
        work_interfere= 2
    else:
        work_interfere= 3
                    
    no_employees = input_dictionary['no_employees']
    if(no_employees =='1-25'):
            no_employees= 1
    elif(no_employees =='26-100'):
            no_employees= 2
    elif(no_employees =='100-500'):
            no_employees= 3
    elif(no_employees =='500-1000'):
            no_employees= 4
    else:
        no_employees= 5
    
    remote_work = input_dictionary['remote_work'] 
    if(remote_work =='No'):
            remote_work= 0
    else:
        remote_work= 1
                        
    tech_company = input_dictionary['tech_company']
    if(tech_company=='No'):
            tech_company= 0
    else:
        tech_company= 1
                        
    benefits = input_dictionary['care_options']
    if(benefits =='No'):
        benefits= 0
    elif(benefits =='Don\'t Know'):
        benefits= 1   
    else:
        benefits= 2
        
    care_options = input_dictionary['care_options']
    if(care_options =='No'):
        care_options= 0
    elif(care_options =='Not sure'):
        care_options= 1   
    else:
        care_options= 2
        
    wellness_program = input_dictionary['wellness_program']
    if(wellness_program =='No'):
        wellness_program= 0
    elif( wellness_program =='Don\'t Know'):
        wellness_program = 1   
    else:
        wellness_program= 2
        
    seek_help = input_dictionary['seek_help'] 
    if(seek_help =='No'):
        seek_help= 0
    elif(input_dictionary['seek_help'] =='Don\'t Know'):
        seek_help= 1   
    else:
        seek_help= 2
    anonymity = input_dictionary['anonymity'] 
    if(anonymity =='No'):
        anonymity= 0
    elif(anonymity =='Don\'t Know'):
        anonymity= 1   
    else:
        anonymity= 2
        
    leave = input_dictionary['leave']
    if(leave =='Very easy'):
        leave = 0
    elif(leave =='Somewhat easy'):
        leave =  1   
    elif(leave =='Don\'t know'):
        leave = 2   
    elif(leave =='Somewhat difficult'):
        leave = 3  
    else:
        leave = 4
                    
    mental_health_consequence = input_dictionary['mental_health_consequence']
    if(mental_health_consequence =='No'):
        mental_health_consequence= 0
    elif(mental_health_consequence =='Maybe'):
        mental_health_consequence= 1   
    else:
        mental_health_consequence= 2
        
    phys_health_consequence = input_dictionary['phys_health_consequence']
    if(phys_health_consequence =='No'):
        phys_health_consequence= 0
    elif(phys_health_consequence=='Maybe'):
        phys_health_consequence= 1   
    else:
        phys_health_consequence= 2
        
    coworkers = input_dictionary['coworkers']
    if(coworkers =='No'):
        coworkers= 0
    elif(coworkers =='Some of them'):
        coworkers= 1   
    else:
        coworkers= 2
    
    supervisor = input_dictionary['supervisor']
    if(supervisor =='No'):
        supervisor= 0
    elif(supervisor =='Some of them'):
        supervisor= 1   
    else:
        supervisor= 2
        
    mental_health_interview = input_dictionary['mental_health_interview']
    if(mental_health_interview =='No'):
        mental_health_interview= 0
    elif(mental_health_interview =='Maybe'):
        mental_health_interview= 1   
    else:
        mental_health_interview= 2
        
    phys_health_interview = input_dictionary['phys_health_interview']
    if(phys_health_interview =='No'):
        phys_health_interview= 0
    elif(phys_health_interview =='Maybe'):
        phys_health_interview =1   
    else:
        phys_health_interview= 2
          
    mental_vs_physical =  input_dictionary['mental_vs_physical']
    if(mental_vs_physical =='Don\'t Know'):
        mental_vs_physical= 0
    elif(mental_vs_physical =='No'):
        mental_vs_physical= 1   
    else:
        mental_vs_physical= 2
        
    obs_consequence = input_dictionary['obs_consequence'] 
    if(obs_consequence =='No'):
        obs_consequence= 0
    else:
        obs_consequence= 1
    
    
    
    input_list = [[age,gender,self_employed,family_history,work_interfere,no_employees,remote_work,tech_company,benefits,care_options,wellness_program,seek_help,anonymity,leave,mental_health_consequence,phys_health_consequence,coworkers,supervisor,mental_health_interview,phys_health_interview,mental_vs_physical,obs_consequence ]]
    
    input_data_standardized = std_scaler.fit_transform(input_list)

    prediction = loaded_model.predict(input_data_standardized)
    
    if (prediction[0] == 0):
        return 'The person is not taking treatment'
    else:
        return 'The person is taking treatment'
    
    

