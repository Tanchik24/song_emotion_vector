<template>
  <div class="pie-chart">
    <div class="pie">
      <div class="hole">
        <div class="hole__inner"></div>
      </div>
    </div>
    <div class="legend-container">
      <PieChartLegendItem title="1 quadrant - 20%" color="#A84CCA"/>
      <PieChartLegendItem title="1 quadrant - 20%" color="#A84CCA"/>
      <PieChartLegendItem title="1 quadrant - 20%" color="#A84CCA"/>
      <PieChartLegendItem title="1 quadrant - 20%" color="#A84CCA"/>
    </div>
  </div>
</template>
<style lang="scss">
.pie-chart {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: space-between;
  border-radius: 24px;
  background-image: linear-gradient(
      to bottom,
      rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
  );
  align-items: center;
}
.pie {
  margin: 28px;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80%;
  aspect-ratio: 1;
  border-radius: 50%;
  rotate: -30deg;
  background: conic-gradient(
      white 0% 20%,
      #633891 20% 40%,
      #727EBE 40% 85%,
      #A84CCA 85%
  );
}
.hole {
  height: 80%;
  width: 80%;
  border-radius: 50%;
  background-color: white;
  display: flex;
  justify-content: center;
  align-items: center;
  &__inner {
    height: 97%;
    width: 97%;
    border-radius: 50%;
    background-color: #311B4E;
  }
}
.legend-container {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  //height: 100%;
  //justify-content: space-between;
}
</style>
<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import PieChartLegendItem from '@/components/PieChartLegendItem.vue';

@Options({
  components: { PieChartLegendItem },
})
export default class PieCharts extends Vue {}
</script>
