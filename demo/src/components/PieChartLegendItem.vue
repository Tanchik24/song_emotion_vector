<template>
  <div class="legend-item">
    <div class="legend-item__title">{{title}}</div>
    <div class="legend-item__color" :style="{ backgroundColor: color }"></div>
  </div>
</template>
<style lang="scss">
.legend-item {
  width: 100%;
  height: 100%;
  display: flex;
  gap: 28px;
  align-items: center;
  margin-right: 28px;
  //background-color: #42b983;
  &__title {
    font-weight: lighter;
    font-size: 16px;
  }
  &__color {
    width: 15%;
    aspect-ratio: 2/1;
    border-radius: 24px;
  }
}
</style>
<script lang="ts">
import { Options, Vue } from 'vue-class-component';

@Options({
  props: {
    title: {
      type: String,
      required: true,
    },
    color: {
      type: String,
      required: true,
    },
  },
})
export default class PieChartLegendItem extends Vue {}
</script>
