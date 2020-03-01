import os, math
import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

path_mapping = "C:/Users/JKKIM/Desktop/Recommender/text_mining_mapping.txt"
f = open(path_mapping, 'r', encoding='UTF-8-sig')
list_mapping = f.read()
f.close()

list_mapping = list_mapping.split('\n')
df_mapping = pd.DataFrame(np.zeros((int(len(list_mapping) / 5), 4)), columns=['Ranking_in_category', 'Naver_name', 'Place_name', 'Before_update'])
list_mapping_ranking = []
list_mapping_naver = []
list_mapping_trip = []
list_mapping_update = []
for i, i_value in enumerate(list_mapping) :
    if i_value != '' and i_value[:4] != 'http' and i % 5 == 0 :
        partial_mapping = i_value.split(' ', 1)
        list_mapping_ranking.append(int(partial_mapping[0].replace('.', '').split('-')[1]))
        partial_mapping = partial_mapping[1:][0]
        if ' **' in partial_mapping :
            partial_mapping = partial_mapping.replace(' **', '')
            list_mapping_update.append(1)
        else :
            list_mapping_update.append(0)
        list_mapping_naver.append(partial_mapping)
    elif i_value != '' and i_value[:4] != 'http' :
        list_mapping_trip.append(i_value)
del partial_mapping
df_mapping['Ranking_in_category'] = list_mapping_ranking
df_mapping['Naver_name'] = list_mapping_naver
df_mapping['Place_name'] = list_mapping_trip
df_mapping['Before_update'] = list_mapping_update   
# 0인 것은 영수증+예약자 전부 있는거. 1과 2는 영수증 리뷰만 있는 것!
df_mapping.iloc[126:, 3] = 2 

        
path_trip = "C:/Users/JKKIM/Desktop/Recommender/crawling_review/"
path_naver = "C:/Users/JKKIM/Desktop/Recommender/crawling_naver/"
list_trip = os.listdir(path_trip)
list_naver = os.listdir(path_naver)
print("해당 경로(path_trip) 내에 존재하는 모든 종류의 파일의 개수 : %d개" % len(list_trip))
print("해당 경로(path_naver) 내에 존재하는 모든 종류의 파일의 개수 : %d개" % len(list_naver))


## a, the 같은 관사들을 stop_words에 추가
stop_words = set(stopwords.words('english')) 



## Trip Advisor
total_place_trip = pd.DataFrame(np.zeros((0, 9)), columns=['Place_name', 'Category', 'Total_rating', 'Total_review_num',
       'Korean_review_num', 'Ranking_in_total', 'Ranking_in_category', 'Address', 'Price'])
total_review_trip = pd.DataFrame(np.zeros((0, 20)), columns=['Place_name', 'ID', 'Location', 'Prev_review', 'Prev_good', 'Rating',
       'Review_date', 'Title', 'Review', 'Visit_date', 'Good', 'Mobile', 'Picture', 'Level', 'Registration', 'Age', 'Sex', 'Visited_city_num',
       'Rating_avg', 'Prev_picture_total'])

for i in range(len(list_trip)) :
    #print("\n# %d번째 파일 이름 : %s" % (i + 1, list_trip[i]))  삼육공 삼육공
    #i = 220
    path_trip2 = path_trip + list_trip[i]
    if list_trip[i][0] == 'p' :
        partial_place_trip = pd.read_csv(path_trip2, encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
        partial_place_trip.iloc[0, 6] = df_mapping[df_mapping['Place_name'] == str(partial_place_trip.iloc[0, 0])].iloc[0, 0]
        total_place_trip = pd.concat([total_place_trip, partial_place_trip])
    elif list_trip[i][0] == 'r' :
        try :
            partial_review_trip = pd.read_csv(path_trip2, encoding='utf-8').drop(['Unnamed: 0'], axis=1)
        except :
            partial_review_trip = pd.read_csv(path_trip2, encoding='utf-16').drop(['Unnamed: 0'], axis=1)
        total_review_trip = pd.concat([total_review_trip, partial_review_trip])
    
total_place_trip.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/trip_place.csv", encoding='utf-8-sig')
total_review_trip.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/trip_review.csv", encoding='utf-8-sig')




## Naver
total_booking = pd.DataFrame(np.zeros((0, 9)), columns=['Rating', 'Booking_review', 'ID', 'Restaurant_reply', 'Reply_info',
       'Image', 'Image_num', 'Restaurant_name', 'Review_date'])
total_place_naver = pd.DataFrame(np.zeros((0, 4)), columns=['Restaurant_name', 'Booking_review_total_rating', 'Booking_review_num',
       'Receipt_review_num'])
total_receipt = pd.DataFrame(np.zeros((0, 14)), columns=['Image', 'Image_num', 'Rating', 'Booking_review', 'ID',
       'previous_review', 'previous_photo', 'previous_receipt_review', 'previous_booking_review', 'previous_receipt_avg',
       'previous_booking_avg', 'Restaurant_name', 'Review_date', 'Visited_frequency'])
total_t_history = pd.DataFrame(np.zeros((0, 7)), columns=['store_name', 'review_type', 'rating', 'location', 'img',
       'restaurant_name', 'ID'])

for i in range(len(list_naver)) :
    #print("\n# %d번째 파일 이름 : %s" % (i + 1, list_trip[i]))  삼육공 삼육공
    #i = 770
    path_naver2 = path_naver + list_naver[i]
    if list_naver[i][0] == 'b' :
        partial_booking = pd.read_csv(path_naver2, encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
        total_booking = pd.concat([total_booking, partial_booking])
    elif list_naver[i][0] == 'p' :
        partial_place_naver = pd.read_csv(path_naver2, encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
        total_place_naver = pd.concat([total_place_naver, partial_place_naver])
    elif list_naver[i][0] == 'r' :
        partial_receipt = pd.read_csv(path_naver2, encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
        total_receipt = pd.concat([total_receipt, partial_receipt])
    elif list_naver[i][0] == 't' :
        partial_t_history = pd.read_csv(path_naver2, encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
        total_t_history = pd.concat([total_t_history, partial_t_history])
    else :
        print("예외상황")

total_booking.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_booking.csv", encoding='utf-8-sig')
total_place_naver.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_place_naver.csv", encoding='utf-8-sig')
total_receipt.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_receipt.csv", encoding='utf-8-sig')
total_t_history.to_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_t_history.csv", encoding='utf-8-sig')




## Read CSV
total_place_trip = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/trip_place.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
total_review_trip = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/trip_review2.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
total_booking = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_booking.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
total_place_naver = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_place_naver.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
total_receipt = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_receipt.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)
total_t_history = pd.read_csv("C:/Users/JKKIM/Desktop/Recommender/crawling_total/naver_t_history.csv", encoding='utf-8-sig').drop(['Unnamed: 0'], axis=1)


## Naver 예약자(Booking) 리뷰 기초통계량
total_booking['Restaurant_name'] = total_booking['Restaurant_name'].astype('str')
total_booking['Restaurant_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_booking['Restaurant_name']]
d_booking = total_booking.copy()
for i in range(len(d_booking)) :
    if '*' in d_booking.iloc[i, 8] :
        cnt_star = d_booking.iloc[i, 8].count('*')
        d_booking.iloc[i, 2] = d_booking.iloc[i, 8].split('*')[0] + cnt_star * '*'
        d_booking.iloc[i, 8] = d_booking.iloc[i, 8].split('*')[-1]
d_booking = d_booking[(pd.to_datetime(d_booking['Review_date'])<= '2020-01-29')]
    
d_booking['Review_count'] = [len(str(d_booking.iloc[i, 1]).rstrip()) if str(d_booking.iloc[i, 1]) != 'nan' else 0 for i in range(len(d_booking))]
d_booking['Reply_count'] = [len(str(d_booking.iloc[i, 3]).rstrip()) if str(d_booking.iloc[i, 3]) != 'nan' else 0 for i in range(len(d_booking))]
d_booking['Image_num'] = d_booking['Image_num'].fillna(0)
d_booking2 = d_booking.drop(columns=['Booking_review', 'ID', 'Image', 'Restaurant_reply', 'Restaurant_name', 'Review_date', 'Reply_info'], axis=1)
d_booking3 = d_booking2[['Image_num', 'Rating', 'Review_count']].rename(
    columns={'Image_num' : 'Booking_image_num', 'Rating' : 'Booking_rating', 'Review_count' : 'Booking_review_count'})

print(d_booking2.describe())
print(d_booking2.cov())
print(d_booking2.corr())
print(d_booking3.describe())
print(d_booking3.cov())
print(d_booking3.corr())


## Naver 영수증(Receipt) 리뷰 기초통계량
total_receipt['Restaurant_name'] = total_receipt['Restaurant_name'].astype('str')
total_receipt['Restaurant_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_receipt['Restaurant_name']]
d_receipt = total_receipt.copy()
d_receipt = d_receipt[(pd.to_datetime(d_receipt['Review_date'])<= '2020-01-29')]
d_receipt['Review_count'] = [len(str(d_receipt.iloc[i, 3]).rstrip()) if str(d_receipt.iloc[i, 3]) != 'nan' else 0 for i in range(len(d_receipt))]
d_receipt['Image_num'] = d_receipt['Image_num'].fillna(0)
d_receipt2 = d_receipt[['Image_num', 'Rating', 'previous_review', 'previous_photo', 'previous_receipt_review', 'previous_booking_review',
                        'previous_receipt_avg', 'previous_booking_avg', 'Visited_frequency', 'Review_count']]
d_receipt3 = d_receipt[['Image_num', 'Rating', 'Review_count']].rename(
    columns={'Image_num' : 'Receipt_image_num', 'Rating' : 'Receipt_rating', 'Review_count' : 'Receipt_review_count'})

d_receipt2_describe = d_receipt2.describe()
d_receipt2_cov = d_receipt2.cov()
d_receipt2_corr = d_receipt2.corr()
print(d_receipt3.describe())
print(d_receipt3.cov())
print(d_receipt3.corr())
stats.ttest_ind(d_booking3[['Booking_review_count']], d_receipt3[['Receipt_review_count']])
stats.ttest_ind(d_booking3[['Booking_rating']], d_receipt3[['Receipt_rating']])
stats.ttest_ind(d_booking3[['Booking_image_num']], d_receipt3[['Receipt_image_num']])


## Naver 영수증 리뷰 작성한 사람의 영수증/예약자 리뷰 기록
total_t_history['store_name'] = total_t_history['store_name'].astype('str')
total_t_history['restaurant_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_t_history['restaurant_name']]
total_t_history['ID'] = total_t_history['ID'].astype('str')
total_t_history['ID'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_t_history['ID']]
d_t_history = total_t_history.copy()

d_t_history2 = d_t_history.drop_duplicates()
d_t_receipt = total_receipt.copy()
d_t_receipt = d_t_receipt[(pd.to_datetime(d_t_receipt['Review_date'])<= '2020-01-29')]
d_t_receipt = d_t_receipt[['Restaurant_name', 'ID']].rename(columns={'Restaurant_name' : 'restaurant_name'})
d_t_set_ID = set(d_t_history.ID.unique()) - set(d_t_receipt.ID.unique())
d_t_set_Name = set(d_t_history.restaurant_name.unique()) - set(d_t_receipt.restaurant_name.unique())
d_t_history2 = d_t_history2[d_t_history['ID'].isin(list(d_t_set_ID)) == False].reset_index().drop(['index'], axis=1)
d_t_history3 = d_t_history2[['rating', 'img']]
print(d_t_history3.describe())

d_t_history4 = pd.merge(d_t_history2, df_mapping[['Naver_name', 'Before_update']], 
                        left_on='restaurant_name', right_on='Naver_name', how='inner').drop(columns=['Naver_name'], axis=1)
d_t_history4 = d_t_history4[d_t_history4['Before_update'] == 0]


d_t_history_receipt = d_t_history4[d_t_history4.review_type == 'receipt_review']
d_t_history_booking = d_t_history4[d_t_history4.review_type == 'booking_review']
print(d_t_history_receipt.describe())
print(d_t_history_booking.describe())


stats.ttest_ind(d_t_history_receipt[['rating']], d_t_history_booking[['rating']])
stats.ttest_ind(d_t_history_receipt[['img']], d_t_history_booking[['img']])

d_t_history_receipt_group = d_t_history_receipt.groupby(['ID'])['store_name'].count().reset_index()
d_t_history_booking_group = d_t_history_booking.groupby(['ID'])['store_name'].count().reset_index()
print(d_t_history_receipt_group.describe())
print(d_t_history_booking_group.describe())

print(len(d_t_history2.store_name.unique()))
d_t_history_receipt_group2 = d_t_history_receipt_group[d_t_history_receipt_group.store_name <= np.percentile(d_t_history_receipt_group.store_name, 25)]
d_t_history_receipt_group3 = d_t_history_receipt_group[d_t_history_receipt_group.store_name <= np.percentile(d_t_history_receipt_group.store_name, 50)]
d_t_history_receipt_group3 = d_t_history_receipt_group3[d_t_history_receipt_group3.store_name > np.percentile(d_t_history_receipt_group3.store_name, 25)]
d_t_history_receipt_group4 = d_t_history_receipt_group[d_t_history_receipt_group.store_name <= np.percentile(d_t_history_receipt_group.store_name, 75)]
d_t_history_receipt_group4 = d_t_history_receipt_group4[d_t_history_receipt_group4.store_name > np.percentile(d_t_history_receipt_group4.store_name, 50)]
d_t_history_receipt_group5 = d_t_history_receipt_group[d_t_history_receipt_group.store_name <= np.percentile(d_t_history_receipt_group.store_name, 100)]
d_t_history_receipt_group5 = d_t_history_receipt_group5[d_t_history_receipt_group5.store_name > np.percentile(d_t_history_receipt_group5.store_name, 75)]
#d_t_history_booking_group2 = d_t_history_booking_group[d_t_history_booking_group.store_name <= np.percentile(d_t_history_booking_group.store_name, 75)]


d_t_history_receipt2 = pd.merge(d_t_history_receipt, 
                                d_t_history_receipt_group2.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
d_t_history_booking2 = pd.merge(d_t_history_booking, 
                                d_t_history_receipt_group2.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
print(d_t_history_receipt2.describe())
print(d_t_history_booking2.describe())
stats.ttest_ind(d_t_history_receipt2[['rating']], d_t_history_booking2[['rating']])
stats.ttest_ind(d_t_history_receipt2[['img']], d_t_history_booking2[['img']])



d_t_history_receipt3 = pd.merge(d_t_history_receipt, 
                                d_t_history_receipt_group3.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
d_t_history_booking3 = pd.merge(d_t_history_booking, 
                                d_t_history_receipt_group3.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
print(d_t_history_receipt3.describe())
print(d_t_history_booking3.describe())
stats.ttest_ind(d_t_history_receipt3[['rating']], d_t_history_booking3[['rating']])
stats.ttest_ind(d_t_history_receipt3[['img']], d_t_history_booking3[['img']])



d_t_history_receipt4 = pd.merge(d_t_history_receipt, 
                                d_t_history_receipt_group4.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
d_t_history_booking4 = pd.merge(d_t_history_booking, 
                                d_t_history_receipt_group4.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
print(d_t_history_receipt4.describe())
print(d_t_history_booking4.describe())
stats.ttest_ind(d_t_history_receipt4[['rating']], d_t_history_booking4[['rating']])
stats.ttest_ind(d_t_history_receipt4[['img']], d_t_history_booking4[['img']])



d_t_history_receipt5 = pd.merge(d_t_history_receipt, 
                                d_t_history_receipt_group5.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
d_t_history_booking5 = pd.merge(d_t_history_booking, 
                                d_t_history_receipt_group5.rename(columns={'store_name' : 'store_count'}), 
                                on='ID', how='inner').drop(columns=['Before_update'], axis=1)
print(d_t_history_receipt5.describe())
print(d_t_history_booking5.describe())
stats.ttest_ind(d_t_history_receipt5[['rating']], d_t_history_booking5[['rating']])
stats.ttest_ind(d_t_history_receipt5[['img']], d_t_history_booking5[['img']])













## Naver 음식점(Place) 기초통계량
total_place_naver['Restaurant_name'] = total_place_naver['Restaurant_name'].astype('str')
total_place_naver['Restaurant_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_place_naver['Restaurant_name']]
d_place_naver = total_place_naver.copy()
d_receipt_group = d_receipt.groupby(['Restaurant_name'])['Rating'].mean().reset_index()
d_booking_group = d_booking.groupby(['Restaurant_name'])['Rating'].mean().reset_index()
d_place_naver = pd.merge(d_place_naver, d_receipt_group, how='inner', on='Restaurant_name').rename(
    columns={'Rating' : 'Receipt_rating', 'Booking_review_total_rating' : 'Info_Booking_rating'})
d_place_naver = pd.merge(d_place_naver, d_booking_group, how='inner', on='Restaurant_name').rename(
    columns={'Rating' : 'Booking_rating'})
print(d_place_naver.describe())
print(d_place_naver.columns)


## Trip Advisor 음식점(Place) 기초통계량
total_place_trip['Place_name'] = total_place_trip['Place_name'].astype('str')
total_place_trip['Place_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_place_trip['Place_name']]
d_place_trip = total_place_trip.copy()
print(d_place_trip.Category.unique())
d_place_trip2 = d_place_trip[['Category', 'Total_rating', 'Total_review_num', 'Korean_review_num']]
print(d_place_trip2.describe())

# One-way ANOVA
d_미국 = d_place_trip2[d_place_trip2.Category == '미국']
d_한국 = d_place_trip2[d_place_trip2.Category == '한국']
d_일본 = d_place_trip2[d_place_trip2.Category == '일본']
d_중국 = d_place_trip2[d_place_trip2.Category == '중국']
d_프랑스 = d_place_trip2[d_place_trip2.Category == '프랑스']
d_이탈리 = d_place_trip2[d_place_trip2.Category == '이탈리']

d_trip_category = ols('Total_rating ~ C(Category)', d_place_trip2).fit()
print(anova_lm(d_trip_category))
F_statistic, pVal = stats.f_oneway(d_미국.Total_rating, d_한국.Total_rating, d_일본.Total_rating, 
                                   d_중국.Total_rating, d_프랑스.Total_rating, d_이탈리.Total_rating)
print("F값: %.5f, P값: %.5f" % (F_statistic, pVal))

d_trip_category = ols('Total_review_num ~ C(Category)', d_place_trip2).fit()
print(anova_lm(d_trip_category))

d_trip_category = ols('Korean_review_num ~ C(Category)', d_place_trip2).fit()
print(anova_lm(d_trip_category))


print(stats.ttest_ind(d_place_naver[['Booking_review_num']], d_place_trip[['Total_review_num']]))
print(stats.ttest_ind(d_place_naver[['Receipt_review_num']], d_place_trip[['Total_review_num']]))

print(stats.ttest_ind(d_place_naver[['Booking_review_num']], d_place_trip[['Korean_review_num']]))
print(stats.ttest_ind(d_place_naver[['Receipt_review_num']], d_place_trip[['Korean_review_num']]))

print(stats.ttest_ind(d_place_naver[['Booking_rating']], d_place_trip[['Total_rating']]))
print(stats.ttest_ind(d_place_naver[['Receipt_rating']], d_place_trip[['Total_rating']]))
print(stats.ttest_ind(d_place_naver[['Info_Booking_rating']], d_place_trip[['Total_rating']]))


## Trip Advisor 리뷰 기초통계량
total_review_trip['Place_name'] = total_review_trip['Place_name'].astype('str')
total_review_trip['Place_name'] = [i_value[:-2] if i_value[-2:] == '.0' else i_value for i_value in total_review_trip['Place_name']]
d_review_trip = total_review_trip.copy()

d_review_trip['Review_date'] = [i_index.replace('년 ', '-').replace('월 ', '-').replace('일', '') for i_index in d_review_trip['Review_date']]
d_complete = []
for i_index in d_review_trip['Review_date'] :
    d_partial = i_index.split('-')
    d_word = d_partial[0] + '-'
    if len(d_partial[1]) == 1 :
        d_word = d_word + '0' + d_partial[1] + '-'
    else :
        d_word = d_word + d_partial[1] + '-'
    if len(d_partial[2]) == 1 :
        d_word = d_word + '0' + d_partial[2]
    else :
        d_word = d_word + d_partial[2]
    d_complete.append(d_word)
d_review_trip2 = d_review_trip[(pd.to_datetime(d_review_trip['Review_date'])<= '2020-01-29')]
d_review_trip2['Review_count'] = [len(str(d_review_trip2.iloc[i, 8]).rstrip()) if str(d_review_trip2.iloc[i, 8]) != 'nan' else 0 for i in range(len(d_review_trip2))]
d_review_trip2['Good'] = d_review_trip2['Good'].fillna(0)
d_review_trip2['Picture'] = d_review_trip2['Picture'].fillna(0)
d_review_trip2['Visited_city_num'] = d_review_trip2['Visited_city_num'].fillna(0)
d_review_trip3 = d_review_trip2[['Place_name', 'Rating', 'Good', 'Picture', 'Visited_city_num', 'Review_count']]
d_review_trip3.describe()

stats.ttest_ind(d_review_trip3[['Review_count']], d_receipt3[['Receipt_review_count']])
stats.ttest_ind(d_review_trip3[['Rating']], d_receipt3[['Receipt_rating']])
stats.ttest_ind(d_review_trip3[['Picture']], d_receipt3[['Receipt_image_num']])

stats.ttest_ind(d_review_trip3[['Review_count']], d_booking3[['Booking_review_count']])
stats.ttest_ind(d_review_trip3[['Rating']], d_booking3[['Booking_rating']])
stats.ttest_ind(d_review_trip3[['Picture']], d_booking3[['Booking_image_num']])


stats.ttest_ind(d_booking3[['Booking_review_count']], d_receipt3[['Receipt_review_count']])



