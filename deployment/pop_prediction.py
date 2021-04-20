# Spotify Pop Prediction
# -----------------------


# import packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Import dataframe template, scalar, and model from notebook

with open('X_user.pickle', 'rb') as file:
	X_user = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
	std = pickle.load(file)

with open('rfr_final.pickle', 'rb') as file:
	rfr_final = pickle.load(file)


# gather continuous variable information

X_user['acousticness'][0] = float(input('\nOn a scale from 0-100, what is the song\'s acousticness? '))/100
X_user['danceability'][0] = float(input('\nOn a scale from 0-100, what is the song\'s danceability? '))/100
X_user['duration_ms'][0] = float(input('\nTo the nearest whole number, how many seconds long is the song? '))*1000
X_user['energy'][0] = float(input('\nOn a scale from 0-100, what is the song\'s energy? '))/100
X_user['liveness'][0] = float(input('\nOn a scale from 0-100, what is the song\'s liveness? '))/100
X_user['loudness'][0] = float(input('\nHow many decibals of headroom does your song have? '))*-1
X_user['speechiness'][0] = float(input('\nOn a scale from 0-100, what is the song\'s speechiness? '))/100
X_user['valence'][0] = float(input('\nOn a scale from 0-100, what is the song\'s valence? '))/100
X_user['tempo'][0] = float(input('\nWhat is the song\'s tempo? '))
X_user['year'][0] = int(input('\nWhat year will this song be released? '))

#gather Categorical variable information

explicit = input('\nDoes the song contain explicit content y/[n]? ')
if explicit == 'y':
	X_user['explicit'][0] = 1


major_minor = input('\nIs the song in a major key y/[n]? ')
if major_minor == 'n':
	X_user['mode'][0] = 0


key_columns = X_user.columns[12:]
key_list = ['A']
for key in key_columns:
	key_list.append(key[4:])

print('\n\nKEY LIST\n')
i=0
for key in key_list:
	print(i, ': ', key)
	i += 1

song_key = int(input('\nWhat key is the song in? input corresponding number from list above: ')) - 1
if song_key >= 0:
	X_user[key_columns[song_key]][0] = 1


# scale continous variables and reassemble dataframe
continous_columns = X_user.columns[:10]
category_columns = X_user.columns[10:]

scaled = std.transform(X_user[continous_columns])
cont_df = pd.DataFrame(scaled, columns=continous_columns)

scaled_df = pd.concat([cont_df, X_user[category_columns]], axis=1)


# predict popularity
predicted_pop = rfr_final.predict(scaled_df)
print('\n\n')
print('The song\'s predicted popularity is ', predicted_pop[0], ' out of 100' )