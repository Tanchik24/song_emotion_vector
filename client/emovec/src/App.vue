<template>
  <div id="background"></div>
  <div id="black-layer"></div>
  <router-view/>
</template>

<style lang="scss">
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500&display=swap');

html, body, #app, h1, h2, h3, p {
  margin: 0;
  padding: 0;
}

#app {
  width: 100%;
  height: 100%;
  font-family: 'Mont', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: white;
}

#black-layer {
  position: fixed;
  width: 100%;
  height: 100%;
  opacity: 80;
  background-image: linear-gradient(
      to bottom,
      #160A22, #300B56, #300B56AA, #300B56AA, #300B56, #160A22
  );
  z-index: -1;
}

#background {
  position: fixed;
  transform: scale(1.1);
  width: 100%;
  height: 100%;
  background-image: url("@/assets/images/background.jpeg");
  filter: blur(20px);
  background-size: cover;
  background-position: center;
  z-index: -2;
}

</style>
<script setup lang="ts">
</script>
