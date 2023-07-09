<template>
  <div class="music-item">
    <p class="music-item__number">1</p>
    <img class="music-item__cover" :src="song.artwork" alt="cover">
    <div class="music-item__content">
      <p class="music-item__title">{{ song.title }}</p>
      <p class="music-item__artist">{{ song.artist }}</p>
    </div>
    <p class="music-item__duration">3:50</p>
    <img class="music-item__fav-btn"
         :class="{ 'jump-animation': this.isJumping,
         'music-item__fav-btn_selected': this.isFavourite }"
         :src="this.isFavourite ?
         require('@/assets/icons/colored_star.svg') : require('@/assets/icons/star.svg')"
         alt="Favourite button"
         v-on:click="this.onClickStarButton"/>
  </div>
</template>

<style scoped lang="scss">
.music-item {
  display: flex;
  gap: 28px;
  align-items: center;

  &__cover {
    width: 64px;
    border-radius: 12px;
  }
  &__number {
    text-align: left;
  }
  &__content {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
  }
  &__title {
    text-align: left;
    font-size: 16px;
    font-weight: 700;
  }
  &__artist {
    text-align: left;
    font-size: 12px;
  }
  &__fav-btn {
    width: 24px;
    height: 24px;
    opacity: 80%;
    &:hover {
      cursor: pointer;
      opacity: 100%;
    }
    &_selected {
      opacity: 100%;
      &:hover {
        opacity: 80%;
      }
    }
  }
}
@media (max-width: 512px) {
  .music-item {
    &__number {
      display: none;
    }
}
}
.jump-animation {
  animation: jump 0.5s linear;
}

@keyframes jump {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-8px);
  }
  100% {
    transform: translateY(0);
  }
}
</style>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import { Song } from '@/models/Song';

@Options({
  props: {
    song: {
      type: Object as () => Song,
      required: true,
    },
  },
  data() {
    return {
      isJumping: false,
    };
  },
  methods: {
    onClickStarButton() {
      if (this.isFavourite) {
        this.$store.commit('removeFromFavourites', this.song.id);
      } else {
        this.$store.commit('addToFavourites', this.song);
      }
      this.isJumping = true;
      setTimeout(() => {
        this.isJumping = false;
      }, 500);
    },
  },
  computed: {
    isFavourite(): boolean {
      const favourites = this.$store.state.favourites as Song[];
      return favourites.some((song) => song.id === this.song.id);
    },
  },
})
export default class MusicCard extends Vue {}
</script>
