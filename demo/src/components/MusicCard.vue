<template>
  <div class="music-card" :style="{backgroundImage: `url(${song.artwork})`}">
    <div class="music-card__overlay"></div>
    <div class="music-card__content">
      <div class="music-card__text-content">
        <div class="music-card__explore-btn">
          <img class="music-card__circle" src="@/assets/icons/circle.svg" alt="circle">
          explore
          <img class="music-card__arrow" src="@/assets/icons/arrow.svg" alt="arrow">
        </div>
        <div class="music-card__meta">
          <p class="music-card__title">{{ song.title }}</p>
          <p class="music-card__artist">{{ song.artist }}</p>
        </div>
      </div>
      <img class="music-card__play-btn" src="@/assets/icons/play.svg" alt="butn">
    </div>
  </div>
</template>

<style scoped lang="scss">

.music-card {
  position: relative;
  width: 360px;
  height: 196px;
  //background-image: url("@/assets/images/serenity.jpg");
  background-size: cover;
  background-position: center;
  border-radius: 32px;
  &__overlay {
    width: 100%;
    height: 100%;
    position: absolute;
    border-radius: 32px;
    top: 0;
    left: 0;
    background-image: linear-gradient(
        to bottom,
        transparent 0%, rgba(16, 5, 27, 0.22) 10%, rgba(21, 5, 37, 0.9) 100%
    );
  }
  &__content {
    display: flex;
    width: 100%;
    height: 100%;
    position: absolute;
    align-items: flex-end;
    top: 0;
    left: 0;
  }
  &__meta {
    display: flex;
    flex-direction: column;
    row-gap: 4px;
    margin: 24px;
  }
  &__title {
    font-size: 16px;
    text-align: left;
    font-weight: 700;
  }
  &__artist {
    text-align: left;
    font-size: 12px;
  }
  &__explore-btn {
    display: flex;
    align-items: center;
    flex-direction: row;
    gap: 10px;
    text-align: left;
    margin: 24px;
    font-size: 12px;
  }
  &__text-content {
    display: flex;
    width: 100%;
    height: 100%;
    flex-direction: column;
    justify-content: space-between;
    //background-color: #300B56;
  }
  &__circle {
    width: 16px;
    height: 16px;
  }
  &__arrow {
    width: 8px;
    height: 8px;
  }
  &__play-btn {
    width: 44px;
    position: absolute;
    bottom: -20px;
    right: 20px;
  }
}
</style>
<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import { Song } from '@/models/Song';

@Options({
  components: {
  },
  props: {
    song: {
      type: Object as () => Song,
      required: true,
    },
  },
})
export default class MusicCard extends Vue {}
</script>
