import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyoff
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from scipy.stats import pearsonr

from PIL import Image
import os

import warnings # Ignores any warning
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


st.set_page_config(
    page_title="Air Quality Index Analysis and Correlation Effect",
    page_icon="👋",
    layout="wide"
)

# Side Bar

with st.sidebar:
    st.title('Capstone Project Tetris Dashboard')
    
    # Penjelasan Partikel Polusi
    with st.expander('Apa itu AQI?'):
        st.write("""
        **The air quality index (AQI)** is an index for reporting air quality on a daily basis. It is a measure of how air pollution affects one's health within a short time period. 
        
        The purpose of the AQI is to help people know how the local air quality impacts their health. The Environmental Protection Agency (EPA) calculates the AQI for five major air pollutants, for which national air quality standards have been established to safeguard public health.
        
        1. Ground-level ozone
        2. Particle pollution/particulate matter (PM2.5/pm 10)
        3. Carbon Monoxide
        4. Sulfur dioxide
        5. Nitrogen dioxide
        
        The higher the AQI value, the greater the level of air pollution and the greater the health concerns. The concept of AQI has been widely used in many developed countries for over the last three decades. AQI quickly disseminates air quality information in real-time.
                
        """)
    
    with st.expander('Apa itu PM10?'):
        st.write('Menurut [BMKG](https://www.bmkg.go.id/kualitas-udara/informasi-partikulat-pm10.bmkg#:~:text=Partikulat%20(PM10)%20adalah%20Partikel,%3D%20150%20%C2%B5gram%2Fm3.), PM10 merupakan partikel udara berukuran 10 Mikrometer atau lebih kecil yang biasa ditemui pada debu dan asap.')
    
    with st.expander('Apa itu SO2?'):
        st.write('[Gas SO2](https://dspace.uii.ac.id/handle/123456789/30912) adalah partikel gas polutan akibat pembakaran bahan bakar fosil seperti minyak yang mampu mengganggu sistem pernapasan manusia.')
    
    with st.expander('Apa itu CO?'):
        st.write('[CO](https://www.alodokter.com/keracunan-karbon-monoksida) (Karbon Monoksida) adalah suatu gas yang timbul akibat asap hasil pembakaran bahan bakar kendaraan bermotor yang berlebihan.')
    
    with st.expander('Apa itu O3?'):
        st.write('[O3](https://dlhk.jogjaprov.go.id/perlindungan-lapisan-ozon) (Ozon) merupakan lapisan molekul gas yang berfungsi untuk menyerap radiasi sinar ultraviolet yang berada di atmosfer bumi.')
    
    with st.expander('Apa itu NO2?'):
        st.write('[NO2](https://pengen-tau.weebly.com/nitrogen-oksida.html) (Nitrogen Dioksida) merupakan gas polutan yang apabila memiliki kadara polusi yang tinggi, maka akan mengganggu paru - paru manusia.')
    
    with st.expander('Data Source'):
        st.markdown("""
        - [x] [AIR QUALITY INDEX (by cities)](https://www.kaggle.com/datasets/ramjasmaurya/most-polluted-cities-and-countries-iqair-index)
        - [x] [AIR QUALITY INDEX (top countries)](https://www.kaggle.com/datasets/ramjasmaurya/most-polluted-cities-and-countries-iqair-index)
        - [x] [Pollutant Standards Index Jogja 2020](https://www.kaggle.com/datasets/adhang/air-quality-in-yogyakarta-indonesia-2020)
        - [x] [GDP Per Capita](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)
        - [x] [CO2 Emissions Per Capita](https://data.worldbank.org/indicator/EN.ATM.CO2E.PC)
        - [x] [Electricity Generated Year](https://github.com/owid/energy-data)
        - [x] [Annual CO Emissions by Region](https://carbonpricingdashboard.worldbank.org/map_data)
        - [x] [Perkembangan Jumlah Kendaraan Bermotor Menurut Jenis (Unit), 2018-2020](https://www.bps.go.id/indicator/17/57/1/jumlah-kendaraan-bermotor.html)
        - [x] [Jumlah Kendaraan Bermotor Menurut Provinsi dan Jenis Kendaraan (unit)](https://www.bps.go.id/indikator/indikator/view_data_pub/0000/api_pub/V2w4dFkwdFNLNU5mSE95Und2UDRMQT09/da_10/1)
        - [x] [Jumlah Penduduk Hasil Proyeksi Menurut Provinsi dan Jenis Kelamin (Ribu Jiwa), 2018-2020](https://www.bps.go.id/indicator/12/1886/1/jumlah-penduduk-hasil-proyeksi-menurut-provinsi-dan-jenis-kelamin.html)
                
        """)


# ------------------------------------

####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 3.3, .05, 2.5, .1))
with row0_1:
    st.title('Air Quality Index (AQI) Analysis 🎯✅ and Correlation Effect ⛅🍃')

    

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((.1, 3.3, .1, 2.5, .1))
with row1_1:

    st.markdown("<span style='color:orange;font-weight:1000;font-size:30px'>**Air Pollution**</span> is the contamination of air due to the presence of substances in the atmosphere that are harmful to the health of humans and other living beings, or cause damage to the climate or to materials.", unsafe_allow_html=True)

    st.markdown("""
            **There are many different types of air pollutants, such as :**

            * <span style='color:#EE4B2B;font-weight:1000;font-size:18px'>*Gases*</span> (including *ammonia, carbon monoxide, sulfur dioxide, nitrous oxides, methane, carbon dioxide* and *chlorofluorocarbons*),
            ```
            PM10 - Particulate Matter
            SO2 - Sulfur Dioxide
            CO - Carbon Monoxide
            O3 - Ozone
            NO2 - Natrium Dioxide
            CFC - Chlorofluorocarbon
            HC - Hidrokarbon
            Pb - Timah
            CO2 - Carbon Diaoksida
            ```
            * <span style='color:#EE4B2B;font-weight:1000;font-size:18px'>*Particulates*</span> (both organic and inorganic)
            * <span style='color:#EE4B2B;font-weight:1000;font-size:18px'>*Biological Molecules*</span>
            
            <span style='color:orange;font-weight:1000;font-size:19px'>🚨 Air pollution can cause diseases, allergies, and even death to humans 🚨</span>

            It can also cause harm to other living organisms such as **animals and food crops, and may damage the natural environment** (for example, climate change, ozone depletion or habitat degradation) or built environment (for example, acid rain).

            Both **human activity** and **natural processes** can generate air pollution.

            ```
            Berdasarkan laporan Kualitas Udara Dunia IQAir 2021 
            pada Maret 2022, Indonesia menempati peringkat teratas 
            sebagai negara paling polusi di dunia dengan konsentrasi 
            PM2,5 tertinggi yakni 34,3 μg/m3. Indonesia juga mendapatkan 
            peringkat pertama di Asia Tenggara sebagai negara yang berpolusi udara.

            ~ "10 Negara Paling Berpolusi di Dunia, Indonesia Nomor Berapa?" 
            selengkapnya https://www.detik.com/edu/detikpedia/d-6137633/10-negara-paling-berpolusi-di-dunia-indonesia-nomor-berapa.
            ```

            ##### **Is this statement true and relevant according to the data? 🤔**

            #### **Objectives :**
            - Is it true that Indonesia has a very bad air index?
            - What are the critical components that have a high impact on Indonesian air?
            - Should Indonesia implement a Carbon Emissions Tax and its relationship to GDP per capita to support economic growth?
    """, unsafe_allow_html=True)
    st.text("")
    st.markdown("##### ***`PLEASE WAIT A MOMENT FOR THE DATA TO LOAD`***")

with row1_2:
    # Streamlit
    if os.path.dirname(os.getcwd()) == "/app":
        d = "/app/air-quality-index-analysis-and-correlation-effect/Dashboard/"
    # Heroku
    elif os.getcwd() == "/app":
        d = "Dashboard/"
    # Local
    else:
        d = os.getcwd()+"/"
    image = Image.open(d+'images/air-pollution-smoke-emission.jpg')
    st.image(image) # caption='air-pollution-smoke-emission.jpg'

    image = Image.open(d+'images/sources-of-air-pollution.png')
    st.image(image, caption="Source of Air Pollution") # caption='Designed by brgfx / Freepik'

    image = Image.open(d+'images/aqi2.png')
    st.image(image, caption="Air Quality Index") # caption='Designed by brgfx / Freepik'


################
### ANALYSIS ###
################

### DATA EXPLORER ###

# Load Data
import os
# Streamlit
if os.path.dirname(os.getcwd()) == "/app":
    d = "/app/air-quality-index-analysis-and-correlation-effect/"
# Heroku
elif os.getcwd() == "/app":
    d = ""
# Local
else:
    d = os.path.dirname(os.getcwd())+"/"
    
# Clean - AIR QUALITY INDEX (by cities).csv
# df_aqicty = pd.read_csv("https://drive.google.com/uc?id=1V086i1eHdM08nk67F4l2D7_bj-ZJk8PY")
df_aqicty = pd.read_csv(d+"data/Most Polluted Cities and Countries (IQAir Index)/Clean - AIR QUALITY INDEX (by cities).csv")

# Clean - AIR QUALITY INDEX- top countries.csv
# df_aqitpcr = pd.read_csv("http s://drive.google.com/uc?id=11qjUGvAQiqEgfPWW8USMz6rHlARcrL_P")
df_aqitpcr = pd.read_csv(d+"data/Most Polluted Cities and Countries (IQAir Index)/Clean - AIR QUALITY INDEX- top countries.csv")

# Clean - pollutant-standards-index-jogja-2020.csv
# df = pd.read_csv("https://drive.google.com/uc?id=1BpMqLEYmGRuOAIyzEsx_-XXdxF5b_1vR")
df = pd.read_csv(d+"data/Air Quality in Yogyakarta, Indonesia/Clean - pollutant-standards-index-jogja-2020.csv")

# GDPPerCapita.csv
# gdp = pd.read_csv("https://drive.google.com/uc?id=1rBgi4F_R9EhYerayeFvaRaRCPQV6N_IY")
gdp = pd.read_csv(d+"data/Co2 Emissions and Economic/GDPPerCapita.csv")

# CO2EmissionsPerCapita.csv
# co2pc = pd.read_csv("https://drive.google.com/uc?id=136cdpMhIX_WKyg5A_FKAjdDspog_Rc7p")
co2pc = pd.read_csv(d+"data/Co2 Emissions and Economic/CO2EmissionsPerCapita.csv")

# ElectricityGeneratedYear.csv
# elecdt = pd.read_csv("https://drive.google.com/uc?id=1f3LClVnVBSjExxdvsl5wSQBv3jwxF1G-")
elecdt = pd.read_csv(d+"data/Co2 Emissions and Economic/ElectricityGeneratedYear.csv")

# AnnualCOEmissionsbyRegion.csv
# co2ann = pd.read_csv("https://drive.google.com/uc?id=1LDi87mDkdnCkl6DN_CqZw9NZprYQGUch")
co2ann = pd.read_csv(d+"data/Co2 Emissions and Economic/AnnualCOEmissionsbyRegion.csv")

# jumlah_kendaraan_bermotor.csv
# df_kendaraan = pd.read_csv("https://drive.google.com/uc?id=1kSguqLIcFnTgqs2r67W0qLUVi0QWyHvh", sep=";")
df_kendaraan = pd.read_csv(d+"data/Additinal Data/jumlah_kendaraan_bermotor.csv", sep=";")

# jumlah_kendaraan_bermotor_provinsi_jenis.csv
# df_kendaraan_prov = pd.read_csv("https://drive.google.com/uc?id=1qFTbI3xHlvNMxdYDQMs34KG50n7Xrl5Y", sep=";")
df_kendaraan_prov = pd.read_csv(d+"data/Additinal Data/jumlah_kendaraan_bermotor_provinsi_jenis.csv", sep=";")

# jumlah_penduduk_provinsi_jk_all.csv
# df_penduduk_all = pd.read_csv("https://drive.google.com/uc?id=1NZZlMpsApa_VSO75TQfpe4qbQs0CXQeu", sep=";")
df_penduduk_all = pd.read_csv(d+"data/Additinal Data/jumlah_penduduk_provinsi_jk_all.csv", sep=";")

row2_spacer1, row2_1, row2_spacer2 = st.columns((.2, 7.1, .2))
with row2_1:
    st.subheader('Data Source')
    st.markdown("_Source 10 Data : IQAir website,  Dinas Lingkungan Hidup, BPS, World Bank, Ember_")
    
    st.markdown("You can click here to see the raw data first 👇")

    see_data = st.expander('AIR QUALITY INDEX (by cities)')
    with see_data:
        st.dataframe(data=df_aqicty.reset_index(drop=True))
    
    see_data2 = st.expander('AIR QUALITY INDEX (top countries)')
    with see_data2:
        st.dataframe(data=df_aqitpcr.reset_index(drop=True))

    see_data3 = st.expander('Pollutant Standards Index Jogja 2020')
    with see_data3:
        st.dataframe(data=df.reset_index(drop=True))

    see_data4 = st.expander('GDP Per Capita')
    with see_data4:
        st.dataframe(data=gdp.reset_index(drop=True))

    see_data5 = st.expander('CO2 Emissions Per Capita')
    with see_data5:
        st.dataframe(data=co2pc.reset_index(drop=True))

    see_data5 = st.expander('Electricity Generated Year')
    with see_data5:
        st.dataframe(data=elecdt.reset_index(drop=True))

    see_data6 = st.expander('Annual CO Emissions by Region')
    with see_data6:
        st.dataframe(data=co2ann.reset_index(drop=True))

    see_data7 = st.expander('Perkembangan Jumlah Kendaraan Bermotor Menurut Jenis (Unit), 2018-2020')
    with see_data7:
        st.dataframe(data=df_kendaraan.reset_index(drop=True))
    
    see_data8 = st.expander('Jumlah Kendaraan Bermotor Menurut Provinsi dan Jenis Kendaraan (unit)')
    with see_data8:
        st.dataframe(data=df_kendaraan_prov.reset_index(drop=True))
    
    see_data9 = st.expander('Jumlah Penduduk Hasil Proyeksi Menurut Provinsi dan Jenis Kelamin (Ribu Jiwa), 2018-2020')
    with see_data9:
        st.dataframe(data=df_penduduk_all.reset_index(drop=True))
st.text('')



#############################################################
# 01. Most Polluted Cities and Countries (IQAir Index).ipynb
#############################################################

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.subheader('Most Polluted Cities and Countries (IQAir Index)')
    st.markdown("##### _Is it true that Indonesia has a very bad air index?_")
    st.markdown('')

row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3  = st.columns((.2, 4.4, 0.1, 6.4, .2))
with row4_1:
    
    aqi = {
        "index":["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"],
        "category":["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
    }
    aqi_tb = pd.DataFrame(aqi)
    st.table(data=aqi_tb.reset_index(drop=True))

    ### Top 10 Polluted Country In World ###
    top_10_country = df_aqitpcr.head(20).copy()
    top_10_country['Rank_new'] = 21-top_10_country['Rank']
    fig1= px.bar(top_10_country, y='Country/Region', 
                x='Rank_new', color='2021',
                title="Top 20 Polluted Country In World",
                text='2021',
                hover_data={'Rank_new':False, 'Rank':True, 'Population':True},
                height=490)
    fig1.layout.plot_bgcolor = "white"
    fig1.add_vline(
        x=13.1, line_width=3, line_dash="dash", 
        line_color="green", annotation_text="Threshold Good",
        annotation_font_color="black",
    )
    fig1.update_layout(margin=dict(t=40, b=10))
    fig1.update_xaxes(visible=False, showticklabels=False)
    st.plotly_chart(fig1, use_container_width=True)


with row4_2:
    top_10_country = df_aqitpcr.head(17).copy().iloc[11:18,:]
    top_10_country['Rank'] = 18-top_10_country['Rank']

    # Deleting unnecesary Columns
    top_10_country.drop(['Population'], axis=1, inplace=True)

    # Converting wide to long format
    top_10_country = top_10_country.melt(id_vars=['Rank', 'Country/Region'],
                                var_name="Year", 
                                value_name="AQI")
                                
    top_10_country.sort_values(['Rank','Year'], ascending=[False, True], inplace=True)
    top_10_country['AQI'] = top_10_country['AQI'].astype(float)


    # For Plotting purose filling missing values with the backfill process
    top_10_country.fillna(method="bfill", inplace=True)

    fig2= px.line(top_10_country, y='AQI', 
                x='Year',
                color='Country/Region',
                title="Top 11-17 Yearly Air Quality Index", 
                symbol='Country/Region',
                text="AQI",height=550)
    fig2.for_each_trace(lambda t: t.update(textfont_color="black", textposition='top right'))
    fig2.layout.plot_bgcolor = "light grey"
    fig2.add_hrect(y0=23, y1=50,
              annotation_text="Good", annotation_position="top left",
              annotation_font_color="black",
              fillcolor="green", opacity=0.15, line_width=0)
    fig2.update_yaxes(visible=False, showticklabels=False, )
    fig2.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    * It can be seen that **Indonesia** has a **17th Rating** with an AQI (Air Quality Index) value of **34.3** so it is considered **Good**.
    * The value of AQI (Air Quality Index) in Indonesia in **2018** was **42**
    * Then it rose 9.7 in **2019** by **51.7**
    * Down in **2020 and 2021** maybe because of the number of vehicles and the process of activity during Covid-19.
    """)    
   


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])
    
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 6.4, 0.1, 4.4, .2))
with row5_1:
    top_10_country = df_aqitpcr.sort_values(['Population'], ascending=False).head(10).copy()
    top_10_country["text"] = top_10_country["Population"].apply(lambda x: human_format(x))
    fig1= px.bar(top_10_country, 
                x="Country/Region",
                y="Population", 
                color='Country/Region',
                text="text",
                title="Top 10 Population Country In World")
    fig1.layout.plot_bgcolor = "white"
    fig1.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig1, use_container_width=True)
with row5_2:
    st.markdown("""
    The total population in Indonesia ranks 4th:
    * China = 1.4B
    * India = 1.4B
    * USA = 331M
    * **Indonesia 273.5M**

    The population of Indonesia has increased from time to time. The increase in population has a negative impact on the environment. the availability of green land as a source of clean air in urban areas is also reduced due to the many existing green lands being converted as settlements. Thus it can be said that an increase in population can lead to reduced availability of clean air. The reduced availability of clean air can also be caused by air pollution due to motor vehicle fumes.  
    """)    


# Get Indonesia Data

# select city in Indonesia
df_aqicty_indo = df_aqicty.loc[df_aqicty['country'] == 'Indonesia'].reset_index(drop=True)

# select country = Indonesia
df_aqitpcr_indo = df_aqitpcr.loc[df_aqitpcr['Country/Region'] == "Indonesia"].reset_index(drop=True)




#############################################################
# 04. Additional Data.ipynb
#############################################################

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])

row22_spacer1, row22_1, row22_spacer2, row22_2, row22_spacer3  = st.columns((.2, 6.4, 0.1, 6.4, .2))
with row22_1:
    df_kendaraan2 = df_kendaraan.copy()
    df_kendaraan2 = df_kendaraan2.melt(id_vars='Year', value_vars=["Mobil Penumpang", "Mobil Bis", "Mobil Barang", "Sepeda motor", "Jumlah"],
                                    var_name='Jenis', value_name='Jumlah')

    df_kendaraan2["text"] = df_kendaraan2["Jumlah"].apply(lambda x: human_format(x))
    df_kendaraan2["Jenis"] = df_kendaraan2["Jenis"].apply(lambda x: "Total" if x == "Jumlah" else x)

    fig2= px.line(df_kendaraan2, y='Jumlah', 
                x='Year',
                color='Jenis',
                title="Number of Vehicles in Indonesia", 
                symbol='Jenis',
                text="text")

    fig2.for_each_trace(lambda t: t.update(textfont_color="black", textposition='bottom right'))
    fig2.layout.plot_bgcolor = "light grey"
    fig2.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)
with row22_2:
    df_kendaraan_prov2 = df_kendaraan_prov[["Year", "Province", "Jumlah"]].copy()
    df_kendaraan_prov2 = df_kendaraan_prov2.sort_values("Jumlah", ascending=False).reset_index(drop=True)
    df_kendaraan_prov2 = df_kendaraan_prov2[(df_kendaraan_prov2["Province"] != "Indonesia") & (df_kendaraan_prov2["Year"] == 2021)].reset_index(drop=True)
    df_kendaraan_prov2["text"] = df_kendaraan_prov2["Jumlah"].apply(lambda x: human_format(x))

    fig = px.bar(df_kendaraan_prov2.head(10), 
                x = 'Province',
                y = 'Jumlah', 
                labels = {'Province': 'Province'}, 
                color = 'Jumlah', 
                text = 'text',
                title = "Indonesia Vehicles by Province",
                height=470
    )

    # plot background white
    fig.layout.plot_bgcolor = "white"
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)



row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    df_aqicty_indo_bar = df_aqicty_indo.copy()
    labels = df_aqicty_indo_bar['city_only']
    fig = go.Figure(data=[
        go.Bar(name="2021", x=labels, y=df_aqicty_indo_bar['2021'], text=df_aqicty_indo_bar['2021']),
        go.Bar(name="2020", x=labels, y=df_aqicty_indo_bar['2020'], text=df_aqicty_indo_bar['2020']),
        go.Bar(name="2019", x=labels, y=df_aqicty_indo_bar['2019'], text=df_aqicty_indo_bar['2019'])
    ])

    # Change the bar mode
    fig.update_layout(title_text='Indonesia Average Air Quality Index by City (3 Years)')
    # fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
    fig.update_layout(barmode='stack')
    fig.update_layout(margin=dict(t=40, b=10))
    fig.for_each_trace(lambda t: t.update(textfont_color="white"))
    st.plotly_chart(fig, use_container_width=True) 


row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 4.4, 0.1, 4.4, .2))
with row7_1:
    st.markdown("""
        * Overall, air pollution has gone down in the last 4 years being year 2021 to be the lowest 
        * Jakarta has highest average AQI score with 39.2 
        * Indralaya in South Sumatra has lowest score with 4.2 
    """)
with row7_2:
    st.markdown("""
        * 4 out of 5 highest polluted cities is in Java 
        * Lowest 5 polluted cities located in Sumatra and Kalimantan Many cities have around 15-25 AQI score
    """)



row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:

    # line plot with plotly express
    df_aqicty_indo_line = df_aqicty_indo.copy()

    city = row8_1.multiselect('Select the City', options=df_aqicty_indo["City"].unique(), default=["Jakarta, Indonesia", "Surabaya, Indonesia", "Pekanbaru, Indonesia"])

    df_aqicty_indo_line = df_aqicty_indo_line[df_aqicty_indo_line["City"].isin(city)]
    # dropping unused columns
    df_aqicty_indo_line.drop(['2021', '2020', '2019', '2018', '2017', 'country', 'city_only'],
                    axis=1, inplace=True)

    # converting wide to long format
    df_aqicty_indo_line = df_aqicty_indo_line.melt(id_vars = ['Rank', 'City'],
                                var_name = "Month", 
                                value_name = "Air Quality Index")

    # filling missing values with the backfill process
    df_aqicty_indo_line.fillna(method="bfill", inplace=True)

    fig= px.line(df_aqicty_indo_line, y = 'Air Quality Index', 
                labels = { 'Air Quality Index': 'AQI'}, 
                x = 'Month', color = 'City',
                title = "Monthly Air Quality Index 2021",
                text="Air Quality Index")
    fig.for_each_trace(lambda t: t.update(textfont_color="black", textposition='top right'))
    fig.layout.plot_bgcolor = "light grey"
    
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        * Jakarta have overall high AQI with highest on July with 57.2
        * Pontianak has staggering rise of 86.2 AQI on November yet followed by inverse effect in cities like Bandung, Jakarta, Serang, and Jambi
        * Indralaya experienced healthy fall of AQI after March by almost 40 points
    """)    


#############################################################
# 02. Air Quality in Yogyakarta, Indonesia.ipynb
#############################################################

row9_spacer1, row9_1, row9_spacer2 = st.columns((.2, 7.1, .2))
with row9_1:
    st.subheader('Air Quality Category and Critical Component')
    st.markdown("##### _What are the critical components that have a high impact on Indonesian air?_")
    st.markdown('')

categoricals = ['Critical Component',	'Category']

numericals = ['PM10',	'SO2',	'CO',	'O3',	'NO2',	'Max']

df_catagg = df.copy()
df_catagg = df_catagg.groupby(["Category"])[["Category"]].count()
df_catagg.rename(columns={"Category": "count"}, inplace=True)
df_catagg.reset_index(inplace=True)
df_catagg = df_catagg.reset_index(drop=True).rename_axis(None, axis=1)
df_catagg =  df_catagg.sort_values(by="count", ascending=False)

df_critagg = df.copy()
df_critagg = df_critagg.groupby(["Critical Component"])[["Critical Component"]].count()
df_critagg.rename(columns={"Critical Component": "count"}, inplace=True)
df_critagg.reset_index(inplace=True)
df_critagg = df_critagg.reset_index(drop=True).rename_axis(None, axis=1)
df_critagg =  df_critagg.sort_values(by="count", ascending=False)

row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Bar(x=df_catagg["Category"], y=df_catagg["count"], text=df_catagg["count"],
                        marker_color=["green", "orange", "red"]),
                1, 1
    )

    fig.add_trace(
        go.Bar(x=df_critagg["Critical Component"], y=df_critagg["count"], text=df_critagg["count"],
                        marker_color=px.colors.qualitative.G10),
                1, 2
    )

    fig.update_layout(height=300, width=800, title_text="Number of Categories and Critical Components", showlegend=False,)
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    
    fig = go.Figure()

    features = numericals
    for i in range(0, len(features)):

        fig.add_trace(go.Box(
            y=df[features[i]],
            name=features[i],
            boxpoints='suspectedoutliers', # only suspected outliers
            marker=dict(
                color='rgb(8,81,156)',
                outliercolor='rgba(219, 64, 82, 0.6)',
                line=dict(
                    outliercolor='rgba(219, 64, 82, 0.6)',
                    outlierwidth=2)),
            line_color='rgb(8,81,156)'
        ))


    fig.update_layout(title_text="Box Plot Styling Outliers")
    fig.layout.plot_bgcolor = "light grey"
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)   


row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 4.4, 0.1, 4.4, .2))
with row11_1:
    st.markdown("""
        **Observation:**
        * There are 3 air qualities in the data
        * Good air quality is the highest air quality detected in 2020 in Yogyakarta at 80%
        * Seen less/unhealthy air quality, too small in 2020
        * CO, O3 and PM10 dominate the category in terms of critical value
    """) 
with row11_2:
    st.markdown("""
        **Observations:**
         * It can be seen that there are outliers detected in some columns
         * Except for column NO2, because it contains only 0 (zero) data
         * However, this oulier data will not be discarded because we want to know the quality of the air produced
    """)


row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))
with row12_1:
    x_axis_val = row12_1.selectbox('Select the Critical Components', options=numericals)

    plot = px.histogram(df.sort_values("Category"), x=x_axis_val, color="Category", 
                   facet_col="Category",
                   color_discrete_sequence=["green", "orange", "red"],
                   title="Distribution of {} values to Category".format(x_axis_val))

    plot.update_layout(margin=dict(t=60, b=10))
    plot.layout.plot_bgcolor = "light grey"
    st.plotly_chart(plot, use_container_width=True)


# Correlation
st.subheader('Correlation Between Air Particles (Critical)')
    
row13_spacer1, row13_1, row13_2, row13_3, row13_spacer2 = st.columns((.2, 1,3,1, .2))
with row13_1:
    x_axis_val = row13_1.selectbox('Select the X-axis', options=numericals, index=2)
    y_axis_val = row13_1.selectbox('Select the Y-axis', options=numericals, index=1)

with row13_2:
    fig = px.scatter(df.sort_values("Category"), x=x_axis_val, y=y_axis_val, color="Category",
                    color_discrete_sequence=["green", "orange", "red"],
                    title="Correlation Category between {} dan {}".format(x_axis_val, y_axis_val))
    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)

with row13_3:
    corr_part = pearsonr(df[x_axis_val], df[y_axis_val])
    st.markdown('##### Correlation between {} and {} (*Pearson*)'.format(x_axis_val, y_axis_val))
    percent = round(corr_part[0]*100,2)

    if percent > 50:
        percent_status = 'High Correlation'
    elif percent > 30:
        percent_status = 'Medium Correlation'
    else:
        percent_status = 'Low Correlation'

    st.subheader(f'{percent}%')
    st.markdown(f'##### ***{percent_status}***')

row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3  = st.columns((.2, 6.4, 0.1, 4.4, .2))
with row13_1:
    df_corr = df[numericals].corr()
    x = list(df_corr.columns)
    y = list(df_corr.index)
    z = np.array(df_corr)

    fig = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
    fig.update_traces(text=np.around(z, decimals=2), texttemplate="%{text}")
    fig.update_xaxes(side="top")
    fig.update_layout(margin=dict(t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
with row13_2:
    X = df[numericals]
    y = df[["Category"]]

    # Calculating Score
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, y)
    scores = fit.scores_

    ft = pd.DataFrame({
        "category":df[numericals].columns,
        "score": scores
    }).sort_values("score", ascending=False)
    ft["score"] = ft["score"].round(decimals = 2)

    # Plotting the ranks
    fig = px.bar(ft, 
                x="category", y="score", 
                color="category",
                color_discrete_sequence=px.colors.qualitative.G10,
                text="score", 
                title="Score Feature Important")
    
    fig.update_layout(margin=dict(t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)




#############################################################
# 03. Co2 Emissions and Economic.ipynb
#############################################################

row14_spacer1, row14_1, row14_spacer2 = st.columns((.2, 7.1, .2))
with row14_1:
    st.subheader('CO2 Emissions and Economic')
    st.markdown("##### _Should Indonesia implement a Carbon Emissions Tax and its relationship to GDP per capita?_")
    st.markdown("##### _How it decreases carbon tax emission and supports economic growth?_")
    st.markdown('')

# DATA CLEANING
gdp.rename(columns={"Country Name":"Country"}, inplace=True)

co2pc.rename(columns={
    "Value":"Co2_p",
    "Attribute":"Year",
    "Country Name":"Country"
}, inplace=True)

gco_join = gdp.merge(co2pc, how='inner', on=['Country', 'Year'])

energygb = elecdt.groupby("Year").agg({
    "Fossil_Energy":"sum",
    "Nuclear_Energy":"sum",
    "Renewable_Electricity":"sum"
}).reset_index()

energy = pd.melt(energygb, id_vars=['Year'], value_vars=['Fossil_Energy', 'Nuclear_Energy','Renewable_Electricity'],
        var_name='Energy Type', value_name='Energy').sort_values(["Year","Energy Type"]).reset_index(drop=True)

co2ann.rename(columns={'Annual CO2 emissions (zero filled)':"Co2Emissions"}, inplace=True)

co2ann = co2ann.groupby("Year").agg({
    "Co2Emissions":"sum"
}).reset_index()

co2ann = co2ann[co2ann["Year"]>=1900]

# VISUALIZATION

row14_spacer1, row14_1, row14_spacer2, row14_2, row14_spacer3  = st.columns((.2, 6.4, 0.1, 6.4, .2))
with row14_1:
    # Generated Electricity (TW) 1985- 2020
    fig = px.area(energy, x="Year", y="Energy", color="Energy Type", line_group="Energy Type")

    fig.update_layout(title='Generated Electricity (TW) 1985- 2020',
                    xaxis_title='Year',
                    yaxis_title='Generated Electricity (TW)')

    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)

with row14_2:
    # Annual Global Co2 Emissions from fossil fuel 1900-2020
    fig = px.area(co2ann, x="Year", y="Co2Emissions")

    fig.update_layout(title='Annual Global Co2 Emissions from fossil fuel 1900-2020',
                    xaxis_title='Year',
                    yaxis_title='CO2 Emissions')
    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)


row15_spacer1, row15_1, row15_spacer2  = st.columns((.2, 7.1, .2))
with row15_1:
    st.markdown("""
        **Fossil fuel** has been the world’s primary source of **electricity**.

        The majority or 50% of the installed capacity of power plants in Indonesia still comes from fossil energy. The need for electrical energy from year to year continues to increase in line with the increasing rate of economic growth, population, and the development of the industrial sector. 
        
        The use of fossil energy will produce waste in the form of CO2 gas which causes infrared radiation from the earth to return to the earth's surface so that it can cause global warming. In addition, the use of fossil energy as the main source of power generation can also result in depletion of natural resource reserves, such as oil, coal, and gas.
    """)


# Annual GDP Vs Co2 Emissions Per Capita

row16_spacer1, row16_1, row16_spacer2 = st.columns((.2, 7.1, .2))
with row16_1:
    st.markdown('### **Carbon Tax**')
    st.markdown('**Carbon tax** is a tax levied on the burning of carbon-based fuels such as coal, oil and gas. The carbon tax is a core policy created to reduce and eliminate the use of fossil fuels whose burning can damage the climate.')


new_gco_join = gco_join.copy()

# United Kingdom
new_gco_join["Co2_p_uk"] = new_gco_join["Co2_p"]*4000
new_gco_join["Co2_p_uk"] = new_gco_join["Co2_p_uk"].round(decimals = 2)

# Sweden
new_gco_join["Co2_p_sw"] = new_gco_join["Co2_p"]*4500
new_gco_join["Co2_p_sw"] = new_gco_join["Co2_p_sw"].round(decimals = 2)

# Indonesia
new_gco_join["Co2_p_indo"] = new_gco_join["Co2_p"]*2000
new_gco_join["Co2_p_indo"] = new_gco_join["Co2_p_indo"].round(decimals = 2)

# India
new_gco_join["Co2_p_india"] = new_gco_join["Co2_p"]*1000
new_gco_join["Co2_p_india"] = new_gco_join["Co2_p_india"].round(decimals = 2)



row17_spacer1, row17_1, row17_spacer2, row17_2, row17_spacer3  = st.columns((.2, 6.4, 0.1, 6.4, .2))
with row17_1:
    # Annual GDP Vs Co2 Emissions Per Capita In the United Kingdom
    gco_uk = new_gco_join[(new_gco_join["Country"] =="United Kingdom") & (new_gco_join["Year"] > 1990)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gco_uk["Year"], y=gco_uk["Co2_p_uk"],
                        mode='lines+markers',
                        name='CO2/Capita'))
    fig.add_trace(go.Scatter(x=gco_uk["Year"], y=gco_uk["GDP"],
                        mode='lines+markers',
                        name='GDP/Capita'))

    # Edit the layout
    fig.update_layout(title='Annual GDP Vs Co2 Emissions Per Capita In the United Kingdom',
                    xaxis_title='Year',
                    yaxis_title='GDP/Capita')
    fig.add_annotation(x=2003.2, y=36500,
                text="EU ETS, 2005*",
                showarrow=True,arrowcolor="#636363",
                ax=-30,
                ay=-90,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8,
                arrowhead=7)

    fig.add_annotation(x=2013, y=44000,
                text="UK CPS, 2013**",
                showarrow=True,arrowcolor="#636363",
                ax=-30,
                ay=-90,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8,
                arrowhead=7)

    fig.update_layout(
            xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="""
                Year
                <br><br><sup>*The European Union Emissions Trading System (EU ETS) is a  form of Carbon Pricing.</sup>
                <br><sup>** UK Carbon Price Support (CPS) is an additonal form of Carbon Pricing.</sup>
                <br><sup>Correlation Coefficient = -0.889 (Strong Negative Correlation)</sup>
                """
                )
            )
        )

    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)

with row17_2:
    # Annual GDP Vs Co2 Emissions Per Capita in Sweden
    gco_sw = new_gco_join[(new_gco_join["Country"] =="Sweden") & (new_gco_join["Year"] > 1975)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gco_sw["Year"], y=gco_sw["Co2_p_sw"],
                        mode='lines+markers',
                        name='CO2/Capita'))
    fig.add_trace(go.Scatter(x=gco_sw["Year"], y=gco_sw["GDP"],
                        mode='lines+markers',
                        name='GDP/Capita'))

    # Edit the layout
    fig.update_layout(title='Annual GDP Vs Co2 Emissions Per Capita In the Sweden',
                    xaxis_title='Year',
                    yaxis_title='GDP/Capita')

    fig.add_annotation(x=1991, y=28000,
                text="Carbon Tax, 1991*",
                showarrow=True,arrowcolor="#636363",
                ax=0,
                ay=-90,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.8,
                arrowhead=1)

    fig.update_layout(
            xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="""
                Year
                <br><br><sup>*Carbon Tax Implementation Started on 1991
                <br>Correlation Coefficient = -0.829 (Strong Negative Correlation)</sup>
                """
                )
            )
        )

    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)


row18_spacer1, row18_1, row18_spacer2 = st.columns((.2, 7.1, .2))
with row18_1:
    st.markdown("""
    **Summary** : After Implementation of ***Carbon Tax***, *UK* and *Sweden* Experience Increased Economic Activity (GDP) 👍 and Decreased Carbon Emissions 🔻
    
    Some 40 countries and more than 20 cities, states and provinces already use carbon pricing mechanisms, with more planning to implement them in the future.
    https://www.worldbank.org/en/programs/pricing-carbon
    """)



row19_spacer1, row19_1, row19_spacer2 = st.columns((.2, 7.1, .2))
with row19_1:
    st.markdown('### **No Carbon Tax**')


row20_spacer1, row20_1, row20_spacer2, row20_2, row20_spacer3  = st.columns((.2, 6.4, 0.1, 6.4, .2))
with row20_1:
    # Annual GDP Vs Co2 Emissions Per Capita in Indonesia
    gco_indo = new_gco_join[(new_gco_join["Country"] =="Indonesia") & (new_gco_join["Year"] > 1990)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gco_indo["Year"], y=gco_indo["Co2_p_indo"],
                        mode='lines+markers',
                        name='CO2/Capita'))
    fig.add_trace(go.Scatter(x=gco_indo["Year"], y=gco_indo["GDP"],
                        mode='lines+markers',
                        name='GDP/Capita'))

    # Edit the layout
    fig.update_layout(title='Annual GDP Vs Co2 Emissions Per Capita In the Indonesia',
                    xaxis_title='Year',
                    yaxis_title='GDP/Capita')

    fig.update_layout(
            xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="""
                Year
                <br><br><sup>Correlation Coefficient is 0.9001 which proves strong</sup>
                <br><sup>positive correlation  for GDP and Co2 Emissions in Indonesia.</sup>
                """
                )
            )
        )

    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)

with row20_2:
    # Annual GDP Vs Co2 Emissions Per Capita in India
    gco_india = new_gco_join[(new_gco_join["Country"] =="India") & (new_gco_join["Year"] > 1990)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gco_india["Year"], y=gco_india["Co2_p_india"],
                        mode='lines+markers',
                        name='CO2/Capita'))
    fig.add_trace(go.Scatter(x=gco_india["Year"], y=gco_india["GDP"],
                        mode='lines+markers',
                        name='GDP/Capita'))

    # Edit the layout
    fig.update_layout(title='Annual GDP Vs Co2 Emissions Per Capita In the India',
                    xaxis_title='Year',
                    yaxis_title='GDP/Capita')

    fig.update_layout(
            xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="""
                Year
                <br><br><sup>Correlation Coefficient is 0.972 which proves</sup>
                <br><sup>strong positive correlation for GDP and Co2 Emissions in India</sup>
                """
                )
            )
        )

    fig.update_layout(margin=dict(t=60, b=10))
    fig.layout.plot_bgcolor = "light grey"
    st.plotly_chart(fig, use_container_width=True)

row21_spacer1, row21_1, row21_spacer2 = st.columns((.2, 7.1, .2))
with row21_1:
    st.markdown("""
    **Summary** : After seeing the increase in the value of Indonesia and India, Compared to the UK and Sweden which implemented the Carbon Tax, **India and Indonesia's Carbon Emissions increased also related to the increase**
    Putting a price on carbon can encourage low-carbon growth and lower greenhouse gas emissions. Putting a Price Tag on Carbon Reduces Carbon Emission and Supports Economic Growth.
    """)


st.markdown('***')
## Find me and let's connect 
html_string = """## Find me
<p>
  <a href="https://www.linkedin.com/in/nurimammasri/" target="_blank"><img alt="LinkedIn" src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" /></a>  
  <a href="https://www.instagram.com/nurimammasri" target="_blank"><img alt="Instagram" src="https://img.shields.io/badge/instagram-%23E4405F.svg?&style=for-the-badge&logo=instagram&logoColor=white" /></a> 
  <a href="mailto:nurimammasri.01@gmail.com" target="_blank"><img alt="Gmail" src="https://img.shields.io/badge/gmail-D14836?&style=for-the-badge&logo=gmail&logoColor=white"/></a> 
  <a href="https://medium.com/@nurimammasri" target="_blank"><img alt="Medium" src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" /></a>  
  <a href="https://github.com/nurimammasri" target="_blank"><img alt="Github" src="https://img.shields.io/github/followers/nurimammasri?style=social" /></a>  
  
</p>
"""
st.markdown(html_string, unsafe_allow_html=True)
