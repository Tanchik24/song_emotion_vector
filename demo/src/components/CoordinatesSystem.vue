<template>
  <div class="chart-container">
    <div class="graph">
      <div class="value-point-container">
        <div class="value-point">
          <p class="value-point__text">0.23 0.24</p>
          <div class="value-point__circle"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<style lang="scss">
.chart-container {
  display: flex;
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background-image: linear-gradient(
    to bottom,
    rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
  );
  justify-items: center;
}
.graph {
  display: flex;
  width: 100%;
  height: 100%;
  margin: auto auto;
  background-image: url("@/assets/images/graph.svg");
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  justify-content: center;
  align-items: center;
}
.value-point {
  display: flex;
  flex-direction: column;
  gap: 8px;
  position: relative;
  left: calc(100% - 10px);
  top: calc(100% - 100% - 65px);
  &__circle {
    width: 16px;
    height: 16px;
    border: 2px solid white;
    border-radius: 50%;
  }
  &__text {
    display: flex;
    width: auto;
    height: 48px;
    font-size: 16px;
    border-radius: 24px;
    justify-content: center;
    align-items: center;
    padding: 0 32px;
    margin-left: 16px;
    background-image: linear-gradient(
      to bottom,
      rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
    );
  }
  &-container {
    display: flex;
    width: 60%;
    height: 70%;
  }
}
</style>
<script setup lang="ts">

</script>
