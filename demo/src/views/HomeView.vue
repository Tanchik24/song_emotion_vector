<template>
  <div class="home">
    <div class="home__left-side">
      <div class="home__left-side-container">
        <h1>YOUR MUSICAL<br>EXPERIENCE</h1>
        <div class="player">
          <p class="player__title">Moonsorrow - Hiidenpelto including Hapean</p>
          <img class="player__img" src="@/assets/images/sexy.jpg" alt="woman in ecstasy">
          <div class="player__controls-container">
            <div class="player__buttons">
              <img class="player__back" src="@/assets/icons/rewind.svg" alt="rewind">
              <img class="player__play-pause" src="@/assets/icons/play.svg" alt="play">
              <img class="player__forward" src="@/assets/icons/rewind.svg" alt="rewind">
            </div>
            <div class="player__seek-bar">
              <div class="player__seek-bar-progress"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="tabs">
        <h1 class="tab">RECOMMENDED</h1>
        <h1 class="tab tab_selected">LIBRARY</h1>
      </div>
      <div class="music-list-controls">
        <div class="random-button" v-on:click="this.onClickRandomButton">
          <img src="@/assets/icons/mix.svg" alt="mix">
          <p>Random tracks</p>
        </div>
      </div>
      <div class="music-list-container">
        <div class="music-list">
          <MusicItem v-for="song in allSongs" :key="song.id" :song="song"/>
        </div>
      </div>
    </div>
    <div class="home__right-side">
      <MusicCard v-for="song in this.favourites" :key="song.id" :song="song"/>
    </div>
  </div>
</template>

<style lang="scss">
.random-button {
  display: flex;
  align-content: center;
  align-items: center;
  gap: 16px;
  &:hover {
    opacity: 80%;
    cursor: pointer;
  }
}
.tab {
  opacity: 50%;
  &_selected {
    opacity: 100%;
  }
  &:hover {
    scale: 98%;
    cursor: pointer;
  }
}

.music-list-controls {
  display: flex;
  align-content: center;
  align-items: center;
  justify-items: center;
  justify-content: right;
}
.tabs {
  display: flex;
  flex-direction: row;
  gap: 64px;
}

@media (max-width: 512px) {
  .music-list-controls {
    justify-content: left;
    margin-left: 4px;
  }
  .tabs {
    flex-direction: column;
    margin-left: 0px;
    gap: 16px
  }
  .home {
    display: flex;
    flex-direction: column;
    &__left-side {
      order: 2;
    }
    &__left-side-container {
      margin-left: 0px;
    }
    &__right-side {
      order: 1;
    }
  }
  .music-list-container {
    margin-left: 0px;
  }
}

.music-list-container {
  display: flex;
  flex-direction: column;
  gap: 32px;
  width: 100%;
  height: 100%;
.music-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
    width: 100%;
    height: 100%;
  }
}
h1 {
  font-size: 46px;
  font-weight: 700;
  text-align: left;
  grid-area: a;
}
.home {
  display: flex;
  width: 100%;
  justify-content: center;
  &__left-side {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 36px;
    max-width: 1024px;
    padding: 64px;
    &-container {
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 32px;
    }
  }
  &__right-side {
    width: 360px;
    display: flex;
    flex-direction: column;
    padding: 64px;
    gap: 64px;
  }
}
.player {
  display: grid;
  grid-template-areas:
    "a a"
    "b c";
  grid-template-columns: 64px 1fr;
  grid-template-rows: 32px 64px;
  align-items: center;
  column-gap: 32px;
  row-gap: 12px;
  grid-area: b;
  &__controls-container {
    display: flex;
    gap: 16px;
    grid-area: c;
    align-items: center;
  }
  &__buttons {
    display: flex;
    gap: 8px;
  }
  &__img {
    width: 64px;
    height: 64px;
    object-fit: cover;
    grid-area: b;
    border-radius: 12px;
  }
  &__title {
    grid-area: a;
    text-align: left;
    font-size: 16px;
  }
  &__forward {
    rotate: 180deg;
  }
  &__play-pause {
    width: 40px;
    height: 40px;
  }
  &__seek-bar {
    display: flex;
    width: 70%;
    height: 8px;
    border-radius: 4px;
    background-color: rgba(217, 217, 217, 0.06);
    align-items: center;
    padding: 0 2px;
    &-progress {
      width: 40%;
      height: 4px;
      border-radius: 2px;
      background-color: rgba(217, 217, 217, 0.4);
    }
  }
}
</style>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import axios from 'axios';
import MusicCard from '@/components/MusicCard.vue';
import MusicItem from '@/components/MusicItem.vue';
import { Song } from '@/models/Song';

@Options({
  components: {
    MusicCard,
    MusicItem,
  },
  data() {
    return {
      allSongs: [] as Song[],
    };
  },
  methods: {
    onClickRandomButton() {
      this.getRandomTracks();
    },
    getRandomTracks() {
      axios.get('http://127.0.0.1:8001/library')
        .then((response) => {
          this.allSongs = response.data;
        });
    },
  },
  mounted() {
    this.getRandomTracks();
  },
  computed: {
    favourites() {
      return this.$store.state.favourites;
    },
  },
})
export default class HomeView extends Vue {}
</script>
