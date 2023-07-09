<template>
    <div class="main">
      <div class="outer-container">
          <NumericalStat title="STD arousal" value="25"/>
          <NumericalStat title="STD valence" value="34"/>
          <NumericalStat title="QC arousal" value="3"/>
          <NumericalStat title="QC valence" value="2"/>
      </div>
      <div class="pie-chart">
        <PieCharts/>
      </div>
    </div>
</template>
<style lang="scss">
.main {
  width: 100%;
  align-items: center;
  display: flex;
  flex-direction: column;
}
.outer-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

</style>
<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import NumericalStat from '@/components/NumericalStat.vue';
import PieCharts from '@/components/PieCharts.vue';
import GPT from '@/components/GPT.vue';
import CoordinatesSystem from '@/components/CoordinatesSystem.vue';

@Options({
  components: {
    NumericalStat,
    PieCharts,
    GPT,
    CoordinatesSystem,
  },
})
export default class MusicInfo extends Vue {}
</script>
