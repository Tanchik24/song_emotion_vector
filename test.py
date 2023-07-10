import requests

files = {'song': open('/Users/tanchik/song_emotion_vector/data/example/Alla_Pugachyova_Ya_tak_hochu_chtoby_leto_ne_konchalos.mp3','rb')}
values = {'song_length': '180', 'period': '45'}
url = 'http://127.0.0.1:8080/predict'
response = requests.post(url, files=files, data=values)
print(response.text)