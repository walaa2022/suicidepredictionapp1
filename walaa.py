import streamlit as st
import pickle as pkl
import base64
import pandas as pd
import sklearn



st.image ('https://www.chathamsafetynet.org/wp-content/uploads/2021/08/Copy-of-Copy-of-PST-Logo-background-e1628697628477.png')

# front end elements of the web page 
html_temp = """ 
    <div style ="background-color:skyblue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Suicide Prediction App</h1> 
    </div> 
    """
      # display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.subheader('by Dr.Walaa Nasr')

st.write('The goal of this project is to gather data about people that might think of suicide attempts, to predict the suicide rates using Machine Learning algorithms and analyzing them to find correlated factors causing increase in suicide rates globally')

st.title('if you can predict it, you can prevent it')

#take input from user

Age=st.selectbox ("Age",range(18,121,1))
              
Sex = st.radio("Select Gender: ", range (1,3,1))
st.info ('male=1, female=2')              

Race = st.radio("Select Race: ", range (1,7,1))
st.info ('White=1, African_American =2, Hispanic=3, Asian=4, Native_American=5, Other=6')  
         
Education = st.slider("Select education: ", 1, 6,20)

score = st.selectbox(" score of ADHD & MD: ", range(0,200,1))

subs = st.selectbox(" substance abuse: ", range(0,200,1))

legal = st.selectbox(" Legal issues: ", range(0,20,1))

Abuse = st.selectbox(" Abuse history: ",range (0,8,1))


ABUSE = st.info ("No=0, physical=1, Sexual=2, Emotional=3, Physical & Sexual=4, Physical & Emotional=5, Sexual & Emotional=6, Physical & Sexual & Emotional=7")

Non_subst_Dx= st.selectbox("non_substance_diagnosis:",range(0,3,1))

st.info('none=0,one=1,more_than_one=2')

Subst_Dx = st.selectbox("substance diagnosis:", range(0,4,1))

st.info('none=0,one_Substance_related=1, two_substance_related=2, three_or_more_substance_related=3')

st.sidebar.subheader("About App")

st.sidebar.info("This web app is helps doctors to find out whether patients are at a risk of commiting suicidal attempts or not")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether your patient has risk for suicidal attempt")
st.sidebar.info("Don't forget to rate this app")

feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
    st.header("Thank you for rating the app!")
    st.info("Caution: This is just a prediction and can't gaurantee that your patient will not try to suicide.") 

    
# convert inputs to DataFrame

df_new = pd.DataFrame ({'Age': [Age], 'Sex':[Sex], "Race": [Race], 'Education': [Education], 'score':[score], 'subs':[subs], 'legal': [legal], 'Abuse': [Abuse], 'Subst_Dx': [Subst_Dx],'Non_subst_Dx':[Non_subst_Dx] })

# load transformer
transformer = pkl.load(open('transformer.pkl','rb'))
#apply transformer on inputs
x_new = transformer.transform (df_new)

# load model                      
loaded_model = pkl.load(open('log_reg.pkl' ,'rb'))


#predict the output
predictx= loaded_model.predict(x_new)[0]

#file_ = open("kramer_gif.gif", "rb")
#contents = loaded_model.read()
#data_url = base64.b64encode(contents).decode("utf-8")
#loaded_model.close()

if st.button("Predict"): 
    if(predictx==1):
        st.error("Warning! this patient has chances of attempting suicide")
    else:
        st.success("this patient is healthy and is less likely to attempt suicide!")
