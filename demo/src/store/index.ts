import { createStore } from 'vuex';
import { Song } from '@/models/Song';

export default createStore({
  state: {
    favourites: [] as Song[],
  },
  getters: {
  },
  mutations: {
    addToFavourites(state, song: Song) {
      state.favourites.push(song);
    },
    removeFromFavourites(state, id: number) {
      state.favourites = state.favourites.filter((song) => song.id !== id);
    },
  },
  actions: {
  },
  modules: {
  },
});
