<template>
  <div class="chart-container">
    <img class="graph" src="@/assets/images/graph.svg" alt="Graph">
  </div>
</template>

<style>
.chart-container {
  display: flex;
  width: 100%;
  height: 100%;
  border-radius: 24px;
  background-image: linear-gradient(
    to bottom,
    rgba(207, 163, 255, 0.2) 10%, rgba(21, 5, 37, 0.9) 100%
  );
  align-items: r;
  justify-items: right;
}
.graph {
  width: 90%;
  margin: 0 auto;
}
</style>
<script setup lang="ts">
</script>
