<template>
  <div class="numerical-stat">
    <div class="numerical-stat__title">{{ title }}</div>
    <div class="numerical-stat__value">{{ value }}</div>
  </div>
</template>
<style lang="scss">
.numerical-stat {
  width: 100%;
  aspect-ratio: 2/1;
  border-radius: 24px;
  background-image: linear-gradient(
      to bottom,
      rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
  );
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  margin: 2%;
  &__title {
    font-weight: lighter;
    font-size: 20px;
  }
  &__value {
    font-weight: bold;
    font-size: 24px;
  }
}
</style>
<script lang="ts">
export default {
  name: 'NumericalStat',
  props: {
    title: String,
    value: Number,
  },
};
</script>
