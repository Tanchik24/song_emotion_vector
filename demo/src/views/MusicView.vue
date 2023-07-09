<template>
  <div class="info">
    <div class="info__card">
      <div class="info__statistics">
        <NumericalStat title="STD arousal" value="56"/>
        <NumericalStat title="STD valence" value="5"/>
        <NumericalStat title="QC arousal" value="5"/>
        <NumericalStat title="QC valence" value="4"/>
      </div>
      <div class="info__charts">
        <div class="info__left">
          <div class="info__pie">
            <PieCharts />
          </div>
          <div class="info__gpt">
            <GPT />
          </div>
        </div>
        <div class="info__right">
          <CoordinateSystem/>
        </div>
      </div>
    </div>
  </div>
</template>
<style lang="scss">
.info {
  display: flex;
  flex: 1;
  width: 100%;
  aspect-ratio: 1360/740;
  justify-content: center;
  align-items: center;
  height: 100%;
  background-color: blueviolet;
  &__card {
    width: 100%;
    height: 100%;
    max-width: 1366px;
    padding: 64px;
    margin-left: auto;
    margin-right: auto;
  }
  &__statistics {
    display: flex;
    flex-direction: row;
    height: 20%;
    align-items: center;
    gap: 64px;
  }
  &__charts {
    display: flex;
    height: 80%;
  }
  &__left {
    width: 50%;
    display: flex;
    flex-direction: column;
  }
  &__pie {
    width: 100%;
    height: 50%;
    display: flex;
    align-items: center;
  }
  &__gpt {
    width: 100%;
    height: 50%;
    display: flex;
    align-items: center;
  }
  &__right {
    width: 50%;
    display: flex;
    align-items: center;
  }
}
</style>
<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import NumericalStat from '@/components/NumericalStat.vue';
import PieCharts from '@/components/PieCharts.vue';
import GPT from '@/components/GPT.vue';
import CoordinateSystem from '@/components/CoordinatesSystem.vue';

@Options({
  components: {
    NumericalStat,
    PieCharts,
    GPT,
    CoordinateSystem,
  },
})
export default class MusicInfo extends Vue {}
</script>
