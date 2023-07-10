import dataclasses
import os
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from api.webapi.repository import get_library
from api.music_preprocess.ContinuousEmotionRepresentation import ContinuousEmotionRepresentation
from api.music_preprocess.gpt import get_description
from api.webapi.recomendation.song_recomendation import get_recommendation


class Server:
    def __init__(self):
        self.__app = Flask("song-emotion-demo")
        CORS(self.__app, resources={r"/*": {"origins": "[http://127.0.0.1:8080,"
                                                       "http://127.0.0.1:8000,"
                                                       "http://localhost:8080,"
                                                       "http://192.168.100.16:8080/,"
                                                       "http://10.5.10.100:8080]"}})
        self.__register_routes()

    def __register_routes(self):
        @self.__app.route("/public/<path:filename>")
        def serve_static(filename):
            return send_from_directory("public", filename)

        @self.__app.route('/predictValuesDemo', methods=['GET'])
        def get_values():
            filename = request.get_json()['filename']
            emotion_representation = ContinuousEmotionRepresentation(song=filename)
            return dataclasses.asdict(emotion_representation.get_static_prediction_result())

        @self.__app.route('/predict', methods=['POST'])
        def get_prediction():
            print(os.getcwd())
            song_length = int(request.form['song_length'])
            period = int(request.form['period'])
            song = request.files['song']
            emotion_representation = ContinuousEmotionRepresentation(song, song_length, period)
            return dataclasses.asdict(emotion_representation.get_static_prediction_result())

        @self.__app.route('/gpt_description', methods=['GET'])
        def get_gpt_description():
            arousal = request.get_json()['arousal']
            valence = request.get_json()['valence']
            arousal = float(arousal)
            valence = float(valence)
            return dataclasses.asdict(get_description(arousal, valence))

        @self.__app.route('/library', methods=['GET'])
        def get_library_api():
            return get_library()

        @self.__app.route('/recommendation', methods=['GET'])
        def get_songs_recommendation():
            ids = request.get_json()['ids']
            return dataclasses.asdict(get_recommendation(ids))

    def run(self):
        self.__app.run(host="0.0.0.0", port=8005, debug=True, use_reloader=False)
